using System.Numerics;
using System.Text.RegularExpressions;
using Ruri.ShaderTools.Pipeline.Frontend;

namespace Ruri.ShaderTools;

internal static class LegacyDx11TessellationInterfaceLinker
{
    private static readonly Regex Field = new(
        @"^\s*(?<prefix>(?:(?:linear|centroid|sample|noperspective|nointerpolation)\s+)*)(?<type>(?:float|int|uint)[1-4]?)\s+(?<name>\w+)\s*:\s*(?<semantic>[A-Za-z_]\w*)\s*;\s*$",
        RegexOptions.CultureInvariant);

    internal static LegacyDx11InterfaceLinkResult Link(UnitySerializedPass pass,
        Func<string, UnitySerializedSubProgram, byte[]?>? resolve)
    {
        var result = new LegacyDx11InterfaceLinkResult();
        try
        {
            var stages = pass.EnumerateProgramSlots().Where(slot => slot.Program.SubPrograms.Count > 0)
                .ToDictionary(slot => slot.Stage, slot => slot.Program.SubPrograms);
            if (resolve is null || !stages.ContainsKey("Vertex") || !stages.ContainsKey("Hull") || !stages.ContainsKey("Domain")
                || stages.Keys.Any(stage => stage is not ("Vertex" or "Hull" or "Domain" or "Fragment")))
                throw new InvalidDataException("Expected VS/HS/DS with optional PS and the original DXBC for each stage.");

            var layouts = new Dictionary<(string, DxbcSignatureKind), IReadOnlyList<DxbcSignatureElement>>();
            var tessellation = new Dictionary<string, DxbcTessellation>();
            var changes = new List<(UnitySerializedSubProgram Program, string Source)>();
            foreach (var (stage, programs) in stages)
            foreach (var program in programs)
            {
                if (!program.Success || program.SourceLanguage != "hlsl" || string.IsNullOrWhiteSpace(program.SourceCode))
                    throw new InvalidDataException($"Missing recovered HLSL for {stage} blob {program.BlobIndex}.");
                byte[] binary = resolve(stage, program) ?? throw new InvalidDataException($"Missing {stage} DXBC blob {program.BlobIndex}.");
                if (stage is "Hull" or "Domain")
                {
                    var declaration = DxbcTessellation.Read(binary);
                    if (declaration.Stage != (stage == "Hull" ? PipelineStage.TessControl : PipelineStage.TessEvaluation)
                        || (tessellation.TryGetValue(stage, out var other) && declaration != other))
                        throw new InvalidDataException($"Inconsistent {stage} tessellator declarations.");
                    tessellation[stage] = declaration;
                }
                foreach (var kind in Kinds(stage))
                {
                    var signature = DxbcSignature.Read(binary, kind);
                    if (signature.Count == 0 && (stage, kind) != ("Fragment", DxbcSignatureKind.Input))
                        throw new InvalidDataException($"Missing {stage} {kind} signature.");
                    var key = (stage, kind);
                    if (layouts.TryGetValue(key, out var previous) &&
                        !previous.Select(Abi).SequenceEqual(signature.Select(Abi)))
                        throw new InvalidDataException($"Variant-dependent {stage} {kind} signature needs per-family linkage.");
                    layouts[key] = signature;
                    if (stage is "Vertex" or "Fragment")
                        changes.Add((program, RestoreEndpoint(program.SourceCode, signature, stage == "Vertex")));
                }
            }

            if (tessellation["Hull"].Domain != tessellation["Domain"].Domain
                || tessellation["Hull"].OutputControlPoints != tessellation["Domain"].InputControlPoints)
                throw new InvalidDataException("HS and DS disagree about the patch domain or control point count.");
            Edge("Vertex", "Hull");
            Edge("Hull", "Domain");
            Edge("Hull", "Domain", patch: true);
            if (stages.ContainsKey("Fragment")) Edge("Domain", "Fragment");
            foreach (var (program, source) in changes)
            {
                if (program.SourceCode != source) result.ChangedPrograms++;
                program.SourceCode = source;
            }
            result.Success = true;
            result.Status = "TESSELLATION_SOURCE_LINKED_NOT_COMPILED";

            void Edge(string from, string to, bool patch = false)
            {
                var outputs = layouts[(from, patch ? DxbcSignatureKind.PatchConstant : DxbcSignatureKind.Output)];
                var inputs = layouts[(to, patch ? DxbcSignatureKind.PatchConstant : DxbcSignatureKind.Input)];
                foreach (var input in inputs)
                {
                    if (to == "Fragment" && IsRasterizerInput(input)) continue;
                    if (!outputs.Any(output => output.SemanticName.Equals(input.SemanticName, StringComparison.OrdinalIgnoreCase)
                        && output.SemanticIndex == input.SemanticIndex && output.Register == input.Register
                        && output.ComponentType == input.ComponentType && output.SystemValue == input.SystemValue
                        && output.Stream == input.Stream && output.MinPrecision == input.MinPrecision
                        && (output.Mask & input.Mask) == input.Mask))
                        throw new InvalidDataException($"Incompatible {from}/{to} {(patch ? "patch" : "stage")} signature at {Semantic(input)}.");
                }
            }
        }
        catch (InvalidDataException error)
        {
            result.Diagnostic = "UNSUPPORTED_TESSELLATION_INTERFACE: " + error.Message;
        }
        return result;
    }

    private static IEnumerable<DxbcSignatureKind> Kinds(string stage)
    {
        if (stage != "Vertex") yield return DxbcSignatureKind.Input;
        if (stage != "Fragment") yield return DxbcSignatureKind.Output;
        if (stage is "Hull" or "Domain") yield return DxbcSignatureKind.PatchConstant;
    }

    private static string RestoreEndpoint(string source, IReadOnlyList<DxbcSignatureElement> signature, bool output)
    {
        string name = output ? "SPIRV_Cross_Output" : "SPIRV_Cross_Input";
        var structures = Regex.Matches(source, @"\bstruct\s+" + name + @"\s*\{(?<body>[^{}]*)\}\s*;", RegexOptions.CultureInvariant);
        if (structures.Count == 0 && signature.Count == 0) return source;
        if (structures.Count != 1) throw new InvalidDataException($"Missing or ambiguous {name}.");
        var body = structures[0].Groups["body"];
        var fields = new Dictionary<string, Match>(StringComparer.OrdinalIgnoreCase);
        foreach (string line in body.Value.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var match = Field.Match(line);
            if (!match.Success || !fields.TryAdd(match.Groups["semantic"].Value, match))
                throw new InvalidDataException($"Unrecognized {name} declaration: {line}");
        }
        var names = fields.Values.Select(m => m.Groups["name"].Value).ToHashSet(StringComparer.Ordinal);
        var declarations = new List<string>();
        foreach (var element in signature)
        {
            string semantic = Semantic(element);
            string type = ScalarType(element);
            int component = BitOperations.TrailingZeroCount((uint)element.Mask);
            // InterstageSlotAssigner encodes each user varying as register * 4 + first lane.
            string generated = element.SystemValue == 0 ? $"TEXCOORD{(ulong)element.Register * 4 + (uint)component}" : semantic;
            if (!fields.Remove(generated, out var match) && element.SystemValue != 0 && element.SemanticIndex == 0)
                fields.Remove(semantic + "0", out match);
            if (match is not null)
            {
                if (match.Groups["type"].Value != type) throw new InvalidDataException($"Width/type mismatch at {semantic}.");
                declarations.Add($"    {match.Groups["prefix"].Value}{type} {match.Groups["name"].Value} : {semantic};");
            }
            else
            {
                if (output || element.UsedMask != 0)
                    throw new InvalidDataException($"Missing live {name} field for {semantic}.");
                string unused = GeneratedNames.UnmappedRegion + "Input_" + declarations.Count;
                if (!names.Add(unused)) throw new InvalidDataException($"Field name collision at {unused}.");
                string interpolation = element.ComponentType is 1 or 2 && element.SystemValue == 0 ? "nointerpolation " : "";
                declarations.Add($"    {interpolation}{type} {unused} : {semantic};");
            }
        }
        if (fields.Count != 0) throw new InvalidDataException($"No original DXBC signature for {name} field {fields.Keys.First()}.");
        string newline = source.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";
        return source[..body.Index] + newline + string.Join(newline, declarations) + newline + source[(body.Index + body.Length)..];
    }

    private static string ScalarType(DxbcSignatureElement element)
    {
        int width = BitOperations.PopCount((uint)element.Mask);
        if (width == 0 || element.Mask > 15 || element.Stream != 0 || element.MinPrecision != 0
            || (element.Mask >> BitOperations.TrailingZeroCount((uint)element.Mask)) != (1 << width) - 1)
            throw new InvalidDataException($"Unsupported precision, stream or component layout at {Semantic(element)}.");
        string scalar = element.ComponentType switch { 1 => "uint", 2 => "int", 3 => "float",
            _ => throw new InvalidDataException($"Unknown scalar type at {Semantic(element)}.") };
        return width == 1 ? scalar : scalar + width;
    }

    private static string Semantic(DxbcSignatureElement element) =>
        element.SystemValue != 0 && element.SemanticIndex == 0 ? element.SemanticName : element.SemanticName + element.SemanticIndex;

    private static DxbcSignatureElement Abi(DxbcSignatureElement element) =>
        element with { SemanticName = element.SemanticName.ToUpperInvariant(), UsedMask = 0 };

    private static bool IsRasterizerInput(DxbcSignatureElement e) =>
        e.SemanticName.Equals("SV_IsFrontFace", StringComparison.OrdinalIgnoreCase)
        || e.SemanticName.Equals("SV_SampleIndex", StringComparison.OrdinalIgnoreCase)
        || e.SemanticName.Equals("SV_PrimitiveID", StringComparison.OrdinalIgnoreCase);
}
