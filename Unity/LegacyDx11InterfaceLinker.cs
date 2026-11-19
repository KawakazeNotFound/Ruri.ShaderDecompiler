using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Ruri.ShaderTools.Unity.ShaderLab;

namespace Ruri.ShaderTools;

/// <summary>A source-interface operation, not evidence of GPU or Unity validation.</summary>
public sealed class LegacyDx11InterfaceLinkResult
{
    public bool Success { get; internal set; }
    public string Status { get; internal set; } = "PARTIAL";
    public string Diagnostic { get; internal set; } = "";
    public int ChangedPrograms { get; internal set; }
    public int AddedPaddingFields { get; internal set; }
    public int WidenedFields { get; internal set; }
    public int CompatiblePairs { get; internal set; }
    public int PreservedUnpairedPrograms { get; internal set; }
    public bool RequiresPairedCompilation { get; internal set; }
    public int VariantFamilies { get; internal set; }
    public List<LegacyDx11DeadOutputProof> PrunedOutputs { get; } = new();
    public List<LegacyDx11FlatOutputProof> PropagatedFlatOutputs { get; } = new();
    public LegacyDx11GeometryInterfaceLinkResult? GeometryChain { get; internal set; }
}

public sealed class LegacyDx11FlatOutputProof
{
    public uint VertexBlobIndex { get; internal set; }
    public uint? ParameterBlobIndex { get; internal set; }
    public string Semantic { get; internal set; } = "";
    public string ScalarType { get; internal set; } = "";
    public List<uint> CompatibleConsumedPixelBlobIndices { get; internal set; } = new();
    public string Reason { get; internal set; } = "ALL_COMPATIBLE_CONSUMERS_EXPLICIT_NOINTERPOLATION";
}

public sealed class LegacyDx11DeadOutputProof
{
    public uint VertexBlobIndex { get; internal set; }
    public uint? ParameterBlobIndex { get; internal set; }
    public string Semantic { get; internal set; } = "";
    public List<uint> CompatiblePixelBlobIndices { get; internal set; } = new();
    public string Reason { get; internal set; } = "NO_COMPATIBLE_PIXEL_CONSUMER";
}

/// <summary>
/// Explicit LegacyDx11 profile only. Parses the flat SPIRV-Cross entry structs
/// and exact main() glue; leaves all static variables and stage mathematics alone.
/// Changes are transactional per pass. Ambiguous/incompatible/unsupported input
/// returns PARTIAL without changing any SourceCode. No original output is removed
/// without a proof over every compatible pixel variant in this pass.
/// </summary>
public static class LegacyDx11InterfaceLinker
{
    private sealed class Unsupported : Exception { public Unsupported(string message) : base(message) { } }
    private sealed record Edit(int Start, int Length, string Text);
    private sealed record Glue(string Variable, int Start, int Length, string Indent);
    private sealed record Field(string Name, string Semantic, string BaseType, int Width, string Modifiers, string Mode)
    {
        // HLSL's omitted semantic index is zero; leading zero spellings do not
        // create a distinct register semantic. Keep the original declaration.
        public string Key => System ? Semantic.ToUpperInvariant()
            : SemanticStem(Semantic).ToUpperInvariant() + SemanticIndex(Semantic).ToString(CultureInfo.InvariantCulture);
        public bool System => Semantic.StartsWith("SV_", StringComparison.OrdinalIgnoreCase);
        public string Type => BaseType + (Width == 1 ? "" : Width.ToString(CultureInfo.InvariantCulture));
        public string Declaration => (Modifiers.Length == 0 ? "" : Modifiers + " ") + Type + " " + Name + " : " + Semantic + ";";
    }
    private sealed record Region(int Open, int Close);
    private sealed class Model
    {
        public UnitySerializedSubProgram Program = null!;
        public bool Vertex;
        public string Source = "";
        public string Mask = "";
        public Region? Interface;
        public Region Main = null!;
        public string MainArguments = "";
        public int MainArgumentsStart;
        public int MainArgumentsLength;
        public int ReturnOffset;
        public string Newline = "\n";
        public List<Field> Fields = new();
        public Dictionary<string, Glue> GlueByField = new(StringComparer.Ordinal);
        public HashSet<string> Consumed = new(StringComparer.OrdinalIgnoreCase);
        public Dictionary<string, uint> ConsumedLanes = new(StringComparer.OrdinalIgnoreCase);
        public HashSet<ushort> Keywords = new();
        public HashSet<string> Pruned = new(StringComparer.OrdinalIgnoreCase);
    }

    private const string TypePattern = @"(?:min16float|min10float|min16int|min16uint|float|half|uint|int|bool)[1-4]?";
    private static readonly Regex FieldPattern = new(@"^\s*((?:(?:linear|centroid|noperspective|sample|nointerpolation)\s+)*)((" + TypePattern + @"))\s+([A-Za-z_]\w*)\s*:\s*([A-Za-z_]\w*)\s*;\s*$", RegexOptions.CultureInvariant);
    private static readonly Regex StageInputArgumentPattern = new(@"^(?:in\s+)?SPIRV_Cross_Input\s+(?<name>[A-Za-z_]\w*)$", RegexOptions.CultureInvariant);
    private static readonly Regex DirectArgumentPattern = new(@"^(?:in\s+)?(?<mods>(?:(?:linear|centroid|noperspective|sample|nointerpolation)\s+)*)(?<type>" + TypePattern + @")\s+(?<name>[A-Za-z_]\w*)\s*:\s*(?<sem>[A-Za-z_]\w*)$", RegexOptions.CultureInvariant);
    private static readonly Regex ConsumptionSwizzle = new(@"\G\s*\.\s*([xyzw]{1,4}|[rgba]{1,4})(?![A-Za-z_0-9])", RegexOptions.CultureInvariant);

    public static LegacyDx11InterfaceLinkResult Link(UnitySerializedPass pass)
        => Link(pass, null);

    public static LegacyDx11InterfaceLinkResult Link(UnitySerializedPass pass,
        Func<string, UnitySerializedSubProgram, byte[]?>? resolveOriginalBinary)
        => Link(pass, resolveOriginalBinary, null);

    public static LegacyDx11InterfaceLinkResult Link(UnitySerializedPass pass,
        Func<string, UnitySerializedSubProgram, byte[]?>? resolveOriginalBinary, List<string>? keywordNames,
        Func<bool>? verifyCompleteSource = null)
    {
        ArgumentNullException.ThrowIfNull(pass);
        var result = new LegacyDx11InterfaceLinkResult();
        try
        {
            var slots = pass.EnumerateProgramSlots().Where(slot => slot.Program.SubPrograms.Count > 0).ToList();
            if (slots.Count == 0)
            {
                result.Success = true;
                result.Status = "SKIPPED_NO_GPU_PROGRAMS";
                return result;
            }
            if (slots.Any(slot => slot.Stage == "Geometry"))
            {
                // A geometry pass has two distinct interface edges. Never treat
                // its vertex output as if it fed the fragment shader directly.
                var geometry = LegacyDx11GeometryInterfaceLinker.Link(pass, resolveOriginalBinary);
                result.GeometryChain = geometry;
                result.Success = geometry.Success;
                result.Status = geometry.Status;
                result.Diagnostic = geometry.Diagnostic;
                if (geometry.Success)
                {
                    result.ChangedPrograms = geometry.SourceProofs.Count(proof => proof.BeforeSha256 != proof.AfterSha256);
                    result.AddedPaddingFields = geometry.AddedPaddingFields;
                }
                return result;
            }
            if (slots.Any(slot => slot.Stage != "Vertex" && slot.Stage != "Fragment"))
                throw new Unsupported("UNSUPPORTED_STAGE_TOPOLOGY: only a direct Vertex/Fragment pass is supported");
            if (pass.ProgVertex?.SubPrograms.Count > 0 && (pass.ProgFragment?.SubPrograms.Count ?? 0) == 0)
            {
                result.Success = true;
                result.Status = "SKIPPED_VERTEX_ONLY_PASS";
                return result;
            }
            var vertices = pass.ProgVertex?.SubPrograms.Select(program => Parse(program, true)).ToList() ?? new();
            var pixels = pass.ProgFragment?.SubPrograms.Select(program => Parse(program, false)).ToList() ?? new();
            if (vertices.Count == 0 || pixels.Count == 0)
                throw new Unsupported("MISSING_ADJACENT_STAGE");
            ValidateVariants(vertices);
            ValidateVariants(pixels);
            var sharedKeywords = vertices.SelectMany(model => model.Keywords).ToHashSet();
            sharedKeywords.IntersectWith(pixels.SelectMany(model => model.Keywords));
            bool Compatible(Model vertex, Model pixel) => sharedKeywords.All(keyword => vertex.Keywords.Contains(keyword) == pixel.Keywords.Contains(keyword));
            var consumers = vertices.Select(vertex => (Vertex: vertex, Pixels: pixels.Where(pixel => Compatible(vertex, pixel)).ToList()))
                .Where(pair => pair.Pixels.Count > 0).ToList();
            var pairedPixels = consumers.SelectMany(pair => pair.Pixels).ToHashSet();
            int unpaired = vertices.Count - consumers.Count + pixels.Count - pairedPixels.Count;
            var completeSource = new Lazy<bool>(() => keywordNames is not null
                && LegacyDx11VariantRouting.CanCatalogStage(pass.ProgVertex!, keywordNames)
                && LegacyDx11VariantRouting.CanCatalogStage(pass.ProgFragment!, keywordNames)
                && verifyCompleteSource?.Invoke() == true);
            if (consumers.Count == 0 || (unpaired > 0 && !completeSource.Value))
                throw new Unsupported("UNPAIRED_VARIANT: no exact shared-keyword projection partner");
            result.CompatiblePairs = consumers.Sum(pair => pair.Pixels.Count);
            var proofs = new List<LegacyDx11DeadOutputProof>();
            var flatProofs = new List<LegacyDx11FlatOutputProof>();
            foreach (var (vertex, compatiblePixels) in consumers)
            {
                // D3D vertex output declarations do not encode the pixel input's
                // rasterizer interpolation. A separately translated PS may retain
                // explicit Flat while its matching VS field has no qualifier.
                // Restore only a proven, unanimous flat consumer contract. Never
                // override an authored VS qualifier or convert a scalar type.
                PropagateProvenFlatOutputs(vertex, compatiblePixels, flatProofs);
                foreach (Field field in vertex.Fields.Where(field => !field.System))
                {
                    if (!compatiblePixels.Any(pixel => pixel.Consumed.Contains(field.Key)))
                    {
                        vertex.Pruned.Add(field.Key);
                        proofs.Add(new LegacyDx11DeadOutputProof
                        {
                            VertexBlobIndex = vertex.Program.BlobIndex,
                            ParameterBlobIndex = vertex.Program.ParameterBlobIndex,
                            Semantic = field.Semantic,
                            CompatiblePixelBlobIndices = compatiblePixels.Select(pixel => pixel.Program.BlobIndex).ToList(),
                        });
                    }
                }
                foreach (Model pixel in compatiblePixels)
                    ValidatePair(vertex, pixel);
            }
            var models = consumers.Select(pair => pair.Vertex).Concat(pixels.Where(pairedPixels.Contains)).ToList();
            var ordered = BuildLayout(models, out string? conflict);
            var updates = new List<(Model Model, string Source)>();
            int padding = 0, widening = 0, familyCount = 0;
            bool pairedCompilation = unpaired > 0;
            if (ordered is not null)
            {
                foreach (Model model in models)
                    updates.Add((model, Rewrite(model, ordered, ref padding, ref widening)));
            }
            else
            {
                Func<Model, string> familyKey;
                var families = unpaired == 0 && keywordNames is not null
                    ? LegacyDx11VariantRouting.ProveVariantFamilies(keywordNames, pass) : null;
                if (families is not null)
                {
                    familyKey = model => families[model.Program];
                }
                else if (completeSource.Value)
                {
                    // The build filter admits only candidates with both original stages.
                    // Thus each surviving assignment takes its exact keyword branch.
                    familyKey = model => string.Join(",", model.Keywords.Where(sharedKeywords.Contains).Order());
                    pairedCompilation = true;
                }
                else
                {
                    throw new Unsupported(keywordNames is null ? conflict! : conflict + "; shared keyword routing is not proven");
                }
                foreach (var family in models.GroupBy(familyKey))
                {
                    var layout = BuildLayout(family, out string? familyConflict);
                    if (layout is null)
                        throw new Unsupported(familyConflict + "; conflict remains in a proven keyword family");
                    foreach (Model model in family)
                        updates.Add((model, Rewrite(model, layout, ref padding, ref widening)));
                    familyCount++;
                }
            }
            // No metadata is mutated before every program passed all checks.
            foreach (var update in updates)
            {
                if (update.Model.Source != update.Source)
                {
                    update.Model.Program.SourceCode = update.Source;
                    result.ChangedPrograms++;
                }
            }
            result.Success = true;
            result.PreservedUnpairedPrograms = unpaired;
            result.RequiresPairedCompilation = pairedCompilation;
            result.Status = familyCount == 0 ? "SOURCE_LINKED_NOT_COMPILED" : "VARIANT_FAMILIES_LINKED_NOT_COMPILED";
            result.VariantFamilies = familyCount;
            result.AddedPaddingFields = padding;
            result.WidenedFields = widening;
            result.PrunedOutputs.AddRange(proofs);
            result.PropagatedFlatOutputs.AddRange(flatProofs);
        }
        catch (Unsupported exception)
        {
            result.Status = "PARTIAL";
            result.Diagnostic = exception.Message;
        }
        return result;
    }

    private static List<Field>? BuildLayout(IEnumerable<Model> models, out string? conflict)
    {
        conflict = null;
        var union = new Dictionary<string, Field>(StringComparer.OrdinalIgnoreCase);
        foreach (Model model in models)
        foreach (Field field in model.Fields.Where(field => !field.System && !model.Pruned.Contains(field.Key)))
        {
            if (!union.TryGetValue(field.Key, out Field? existing))
                union.Add(field.Key, field);
            else
            {
                if (existing.BaseType != field.BaseType || existing.Mode != field.Mode)
                {
                    conflict = "PASS_UNION_CONFLICT: semantic " + field.Semantic + " has incompatible scalar type or interpolation";
                    return null;
                }
                if (field.Width > existing.Width)
                    union[field.Key] = existing with { Width = field.Width };
            }
        }
        return union.Values.OrderBy(field => SemanticStem(field.Semantic), StringComparer.OrdinalIgnoreCase)
            .ThenBy(field => SemanticIndex(field.Semantic)).ToList();
    }

    private static void PropagateProvenFlatOutputs(Model vertex, List<Model> pixels, List<LegacyDx11FlatOutputProof> proofs)
    {
        for (int index = 0; index < vertex.Fields.Count; index++)
        {
            Field output = vertex.Fields[index];
            if (output.System || output.Modifiers.Length != 0 || output.BaseType is not ("float" or "uint" or "int"))
                continue;
            var consumed = pixels.Where(pixel => pixel.Consumed.Contains(output.Key))
                .Select(pixel => (Pixel: pixel, Input: pixel.Fields.Single(field => field.Key == output.Key))).ToList();
            if (consumed.Count == 0 || consumed.Any(pair => pair.Input.Mode != "nointerpolation" || pair.Input.BaseType != output.BaseType))
                continue;
            vertex.Fields[index] = output with { Modifiers = "nointerpolation", Mode = "nointerpolation" };
            proofs.Add(new LegacyDx11FlatOutputProof
            {
                VertexBlobIndex = vertex.Program.BlobIndex,
                ParameterBlobIndex = vertex.Program.ParameterBlobIndex,
                Semantic = output.Semantic,
                ScalarType = output.BaseType,
                CompatibleConsumedPixelBlobIndices = consumed.Select(pair => pair.Pixel.Program.BlobIndex).ToList(),
            });
        }
    }

    private static void ValidateVariants(List<Model> models)
    {
        var keys = new HashSet<string>(StringComparer.Ordinal);
        foreach (Model model in models)
        {
            string key = string.Join(",", model.Keywords.OrderBy(keyword => keyword));
            // The source writer dispatches by keyword condition only. Tier/API
            // ambiguity cannot be resolved from its defaulted metadata fields.
            if (!keys.Add(key))
                throw new Unsupported("DUPLICATE_VARIANT_CONDITION: source selection is ambiguous");
        }
    }

    private static void ValidatePair(Model vertex, Model pixel)
    {
        foreach (Field input in pixel.Fields)
        {
            if (!pixel.Consumed.Contains(input.Key)) continue;
            if (input.System && IsRasterizerGenerated(input.Semantic)) continue;
            Field? output = vertex.Fields.SingleOrDefault(field => field.Key == input.Key);
            if (output is null || vertex.Pruned.Contains(input.Key))
                throw new Unsupported("MISSING_PRODUCER: consumed pixel semantic " + input.Semantic);
            if (output.BaseType != input.BaseType || output.Mode != input.Mode)
                throw new Unsupported("PAIR_TYPE_CONFLICT: consumed pixel semantic " + input.Semantic);
            uint available = (1u << output.Width) - 1;
            if ((pixel.ConsumedLanes[input.Key] & ~available) != 0)
                throw new Unsupported("MISSING_PRODUCER_LANES: consumed pixel semantic " + input.Semantic
                    + " reads components outside the original vertex output width");
        }
    }

    private static Model Parse(UnitySerializedSubProgram program, bool vertex)
    {
        if (!program.Success || string.IsNullOrWhiteSpace(program.SourceCode) || !string.Equals(program.SourceLanguage, "hlsl", StringComparison.OrdinalIgnoreCase))
            throw new Unsupported("SOURCE_UNAVAILABLE_OR_NOT_HLSL");
        var model = new Model { Program = program, Vertex = vertex, Source = program.SourceCode!, Newline = program.SourceCode!.Contains("\r\n") ? "\r\n" : "\n" };
        model.Mask = MaskTrivia(model.Source);
        model.Keywords = program.KeywordIndices.ToHashSet();
        if (model.Keywords.Count != program.KeywordIndices.Count)
            throw new Unsupported("DUPLICATE_KEYWORD_INDEX");
        string structName = vertex ? "SPIRV_Cross_Output" : "SPIRV_Cross_Input";
        var structMatches = Regex.Matches(model.Mask, @"\bstruct\s+" + structName + @"\s*\{");
        if (structMatches.Count > 1 || (vertex && structMatches.Count != 1))
            throw new Unsupported("AMBIGUOUS_OR_MISSING_ENTRY_STRUCT");
        if (structMatches.Count == 1)
        {
            int open = model.Mask.IndexOf('{', structMatches[0].Index);
            model.Interface = new Region(open, MatchingBrace(model.Mask, open));
            string body = model.Mask[(open + 1)..model.Interface.Close];
            var names = new HashSet<string>(StringComparer.Ordinal);
            var semantics = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (string segment in body.Split(';'))
            {
                if (string.IsNullOrWhiteSpace(segment)) continue;
                Match fieldMatch = FieldPattern.Match(segment + ";");
                if (!fieldMatch.Success)
                    throw new Unsupported("UNSUPPORTED_STRUCT_FIELD: arrays/matrices/nested declarations require a richer IR linker");
                string type = fieldMatch.Groups[2].Value;
                Match typeMatch = Regex.Match(type, @"^(.*?)([1-4]?)$");
                string scalar = typeMatch.Groups[1].Value;
                int width = typeMatch.Groups[2].Value.Length == 0 ? 1 : int.Parse(typeMatch.Groups[2].Value, CultureInfo.InvariantCulture);
                string modifiers = string.Join(" ", fieldMatch.Groups[1].Value.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
                string mode = InterpolationMode(modifiers);
                var field = new Field(fieldMatch.Groups[4].Value, fieldMatch.Groups[5].Value, scalar, width, modifiers, mode);
                if (!names.Add(field.Name) || !semantics.Add(field.Key))
                    throw new Unsupported("DUPLICATE_INTERFACE_FIELD_OR_SEMANTIC");
                ValidateSystem(field, vertex);
                model.Fields.Add(field);
            }
        }
        var mainMatches = Regex.Matches(model.Mask, @"\b(?<ret>SPIRV_Cross_Output|void)\s+main\s*\((?<args>[^()]*)\)\s*\{");
        if (mainMatches.Count != 1)
            throw new Unsupported("UNSUPPORTED_MAIN_SIGNATURE");
        Match main = mainMatches[0];
        string returnType = main.Groups["ret"].Value;
        if (vertex && returnType != "SPIRV_Cross_Output")
            throw new Unsupported("UNSUPPORTED_MAIN_SIGNATURE");
        string args = main.Groups["args"].Value.Trim();
        ValidateMainArguments(args, vertex, model.Interface is not null, model.Fields);
        model.MainArgumentsStart = main.Groups["args"].Index;
        model.MainArgumentsLength = main.Groups["args"].Length;
        model.MainArguments = model.Source.Substring(model.MainArgumentsStart, model.MainArgumentsLength).Trim();
        int mainOpen = model.Mask.IndexOf('{', main.Index);
        model.Main = new Region(mainOpen, MatchingBrace(model.Mask, mainOpen));
        string mainBody = model.Mask[(mainOpen + 1)..model.Main.Close];
        if (mainBody.Contains('{') || mainBody.Contains('}'))
            throw new Unsupported("UNSUPPORTED_NESTED_MAIN_GLUE");
        Match? returned = null;
        if (returnType == "SPIRV_Cross_Output")
        {
            var returns = Regex.Matches(mainBody, @"(?m)^[ \t]*return\s+stage_output\s*;");
            if (returns.Count != 1)
                throw new Unsupported("UNSUPPORTED_MAIN_RETURN");
            returned = returns[0];
            model.ReturnOffset = mainOpen + 1 + returned.Index;
        }
        else if (Regex.IsMatch(mainBody, @"\breturn\b\s*[^;\s]"))
        {
            throw new Unsupported("UNSUPPORTED_MAIN_RETURN");
        }
        string outsideMain = model.Mask[..main.Index] + model.Mask[(model.Main.Close + 1)..];
        foreach (Field field in model.Fields)
        {
            string expression = vertex
                ? @"(?m)^(?<indent>[ \t]*)stage_output\." + Regex.Escape(field.Name) + @"\s*=\s*(?<variable>[A-Za-z_]\w*)\s*;"
                : @"(?m)^(?<indent>[ \t]*)(?<variable>[A-Za-z_]\w*)\s*=\s*stage_input\." + Regex.Escape(field.Name) + @"\s*;";
            var matches = Regex.Matches(mainBody, expression);
            int references = Regex.Matches(mainBody, (vertex ? @"\bstage_output\." : @"\bstage_input\.") + Regex.Escape(field.Name) + @"\b").Count;
            if (matches.Count != 1 || references != 1)
                throw new Unsupported("UNSUPPORTED_ENTRY_GLUE: expected one direct static field assignment");
            Match glue = matches[0];
            string variable = glue.Groups["variable"].Value;
            int declarations = Regex.Matches(outsideMain, @"\bstatic\s+" + Regex.Escape(field.Type) + @"\s+" + Regex.Escape(variable) + @"\s*;").Count;
            if (declarations != 1)
                throw new Unsupported("UNCONFIRMED_STATIC_INTERFACE_TYPE");
            model.GlueByField[field.Name] = new Glue(variable, mainOpen + 1 + glue.Index, glue.Length, glue.Groups["indent"].Value);
            if (!vertex)
            {
                uint consumed = ConsumedLaneMask(model, field, model.GlueByField[field.Name]);
                model.ConsumedLanes.Add(field.Key, consumed);
                if (consumed != 0) model.Consumed.Add(field.Key);
            }
        }
        ValidateModifiedInterfaceUses(model, structName, structMatches, main, returned, mainOpen);
        if (vertex && model.Fields.Count(field => field.Key == "SV_POSITION") != 1)
            throw new Unsupported("MISSING_VERTEX_POSITION");
        return model;
    }

    private static void ValidateMainArguments(string args, bool vertex, bool hasInterface, List<Field> interfaceFields)
    {
        if (args.Length == 0)
        {
            if (!vertex && hasInterface)
                throw new Unsupported("UNSUPPORTED_MAIN_ARGUMENTS");
            return;
        }
        bool sawStageInput = false;
        var names = new HashSet<string>(StringComparer.Ordinal);
        var semantics = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (!vertex)
            foreach (Field field in interfaceFields)
                semantics.Add(field.Key);
        foreach (string rawSegment in args.Split(','))
        {
            string segment = Regex.Replace(rawSegment.Trim(), @"\s+", " ");
            if (segment.Length == 0)
                throw new Unsupported("UNSUPPORTED_MAIN_ARGUMENTS");
            Match stageInput = StageInputArgumentPattern.Match(segment);
            if (stageInput.Success)
            {
                string name = stageInput.Groups["name"].Value;
                if (sawStageInput || !names.Add(name))
                    throw new Unsupported("UNSUPPORTED_MAIN_ARGUMENTS");
                if (vertex ? name == "stage_output" : (!hasInterface || name != "stage_input"))
                    throw new Unsupported("UNSUPPORTED_MAIN_ARGUMENTS");
                sawStageInput = true;
                continue;
            }
            Match direct = DirectArgumentPattern.Match(segment);
            if (!direct.Success)
                throw new Unsupported("UNSUPPORTED_MAIN_ARGUMENTS");
            string paramName = direct.Groups["name"].Value;
            string paramType = direct.Groups["type"].Value;
            string semantic = direct.Groups["sem"].Value;
            if (paramName is "stage_input" or "stage_output" || !names.Add(paramName))
                throw new Unsupported("UNSUPPORTED_MAIN_ARGUMENTS");
            Match typeMatch = Regex.Match(paramType, @"^(.*?)([1-4]?)$");
            string scalar = typeMatch.Groups[1].Value;
            int width = typeMatch.Groups[2].Value.Length == 0 ? 1 : int.Parse(typeMatch.Groups[2].Value, CultureInfo.InvariantCulture);
            string modifiers = string.Join(" ", direct.Groups["mods"].Value.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
            var field = new Field(paramName, semantic, scalar, width, modifiers, InterpolationMode(modifiers));
            if (vertex)
            {
                if (field.System)
                    ValidateVertexInputSystem(field);
                else
                    SemanticIndex(field.Semantic);
            }
            else
            {
                if (!field.System)
                    throw new Unsupported("UNSUPPORTED_MAIN_ARGUMENTS");
                ValidateSystem(field, false);
            }
            if (!semantics.Add(field.Key))
                throw new Unsupported("DUPLICATE_INTERFACE_FIELD_OR_SEMANTIC");
        }
        if (!vertex && hasInterface && !sawStageInput)
            throw new Unsupported("UNSUPPORTED_MAIN_ARGUMENTS");
    }

    private static void ValidateVertexInputSystem(Field field)
    {
        bool valid = field.Key switch
        {
            "SV_VERTEXID" or "SV_INSTANCEID" => field.BaseType == "uint" && field.Width == 1,
            _ => false,
        };
        if (!valid) throw new Unsupported("UNSUPPORTED_SYSTEM_SEMANTIC: " + field.Semantic);
    }

    private static void Clear(char[] mask, int start, int length)
    {
        for (int index = start; index < start + length; index++) mask[index] = ' ';
    }

    private static void ValidateModifiedInterfaceUses(Model model, string structName, MatchCollection structs,
        Match main, Match? returned, int mainOpen)
    {
        // Only the interface being rewritten is restricted. The VS input struct
        // is not modified here, so valid production VS input glue/whole reads stay legal.
        char[] remaining = model.Mask.ToCharArray();
        if (model.Interface is not null)
            Clear(remaining, structs[0].Index, model.Interface.Close + 1 - structs[0].Index);
        if (model.Vertex)
        {
            Clear(remaining, main.Index, "SPIRV_Cross_Output".Length);
            string body = model.Mask[(mainOpen + 1)..model.Main.Close];
            var declaration = Regex.Matches(body, @"\bSPIRV_Cross_Output\s+stage_output\s*;");
            if (declaration.Count != 1)
                throw new Unsupported("UNSUPPORTED_OUTPUT_WRAPPER_DECLARATION");
            Clear(remaining, mainOpen + 1 + declaration[0].Index, declaration[0].Length);
            Clear(remaining, mainOpen + 1 + returned!.Index, returned.Length);
        }
        else
            Clear(remaining, model.MainArgumentsStart, model.MainArgumentsLength);
        foreach (Glue glue in model.GlueByField.Values) Clear(remaining, glue.Start, glue.Length);
        string residue = new(remaining);
        string identifier = model.Vertex ? "stage_output" : "stage_input";
        if (Regex.IsMatch(residue, @"\b" + identifier + @"\b")
            || Regex.IsMatch(residue, @"\b" + structName + @"\b"))
            throw new Unsupported("UNSUPPORTED_INTERFACE_AGGREGATE_USE: modified interface escapes proven field glue");
    }

    private static uint ConsumedLaneMask(Model model, Field field, Glue glue)
    {
        // Remove only the one confirmed static declaration and direct input glue.
        // Remaining uses anywhere (including main) participate in the proof.
        char[] uses = model.Mask.ToCharArray();
        // SPIRV-Cross commonly gives a struct field and its distinct static
        // global the same spelling. The field declaration is not a global read.
        if (model.Interface is not null)
            Clear(uses, model.Interface.Open + 1, model.Interface.Close - model.Interface.Open - 1);
        var declarations = Regex.Matches(model.Mask, @"\bstatic\s+" + Regex.Escape(field.Type)
            + @"\s+" + Regex.Escape(glue.Variable) + @"\s*;");
        if (declarations.Count != 1) throw new Unsupported("UNCONFIRMED_STATIC_INTERFACE_TYPE");
        Clear(uses, declarations[0].Index, declarations[0].Length);
        Clear(uses, glue.Start, glue.Length);
        string source = new(uses);
        uint all = (1u << field.Width) - 1, consumed = 0;
        foreach (Match use in Regex.Matches(source, @"\b" + Regex.Escape(glue.Variable) + @"\b"))
        {
            Match swizzle = ConsumptionSwizzle.Match(source, use.Index + use.Length);
            // Bare values, indexing, argument/alias passing and unrecognized uses
            // conservatively consume the entire original input width.
            if (!swizzle.Success) return all;
            foreach (char component in swizzle.Groups[1].Value)
            {
                int lane = "xyzw".IndexOf(component);
                if (lane < 0) lane = "rgba".IndexOf(component);
                consumed |= 1u << lane;
            }
        }
        if ((consumed & ~all) != 0) throw new Unsupported("UNSUPPORTED_INPUT_SWIZZLE_LANES");
        return consumed;
    }

    private static string Rewrite(Model model, List<Field> union, ref int padding, ref int widening)
    {
        var edits = new List<Edit>();
        var fields = new List<Field>();
        var addAssignments = new StringBuilder();
        foreach (Field canonical in union)
        {
            Field? original = model.Fields.SingleOrDefault(field => field.Key == canonical.Key && !model.Pruned.Contains(field.Key));
            if (original is null)
            {
                string name = "vrcrl_pad_" + canonical.Key;
                if (Regex.IsMatch(model.Mask, @"\b" + Regex.Escape(name) + @"\b"))
                    throw new Unsupported("PADDING_IDENTIFIER_COLLISION");
                fields.Add(canonical with { Name = name });
                padding++;
                if (model.Vertex)
                    addAssignments.Append("    stage_output.").Append(name).Append(" = ").Append(Zero(canonical)).Append(';').Append(model.Newline);
                continue;
            }
            fields.Add(original with { Width = canonical.Width });
            if (canonical.Width == original.Width) continue;
            widening++;
            Glue glue = model.GlueByField[original.Name];
            string statement;
            if (model.Vertex)
            {
                string zeros = string.Join(", ", Enumerable.Repeat(ScalarZero(canonical.BaseType), canonical.Width - original.Width));
                statement = glue.Indent + "stage_output." + original.Name + " = " + canonical.Type + "(" + glue.Variable + ", " + zeros + ");";
            }
            else
                statement = glue.Indent + glue.Variable + " = stage_input." + original.Name + "." + "xyzw"[..original.Width] + ";";
            edits.Add(new Edit(glue.Start, glue.Length, statement));
        }
        // System values stay at the tail, in their own original per-program order.
        // In particular no mono program gains a synthetic render-target-array output.
        fields.AddRange(model.Fields.Where(field => field.System));
        foreach (Field original in model.Fields.Where(field => model.Pruned.Contains(field.Key)))
        {
            Glue glue = model.GlueByField[original.Name];
            edits.Add(new Edit(glue.Start, glue.Length, ""));
        }
        string declarations = model.Newline + string.Join(model.Newline, fields.Select(field => "    " + field.Declaration)) + model.Newline;
        if (model.Interface is not null)
            edits.Add(new Edit(model.Interface.Open + 1, model.Interface.Close - model.Interface.Open - 1, declarations));
        else if (fields.Count > 0)
        {
            edits.Add(new Edit(0, 0, "struct SPIRV_Cross_Input" + model.Newline + "{" + declarations + "};" + model.Newline));
            string updatedArgs = model.MainArguments.Length == 0
                ? "SPIRV_Cross_Input stage_input"
                : "SPIRV_Cross_Input stage_input, " + model.MainArguments;
            edits.Add(new Edit(model.MainArgumentsStart, model.MainArgumentsLength, updatedArgs));
        }
        if (addAssignments.Length > 0)
            edits.Add(new Edit(model.ReturnOffset, 0, addAssignments.ToString()));
        string result = model.Source;
        int previousStart = result.Length + 1;
        foreach (Edit edit in edits.OrderByDescending(edit => edit.Start))
        {
            if (edit.Start + edit.Length > previousStart)
                throw new Unsupported("OVERLAPPING_SOURCE_EDITS");
            result = result.Remove(edit.Start, edit.Length).Insert(edit.Start, edit.Text);
            previousStart = edit.Start;
        }
        return result;
    }

    private static string ScalarZero(string type) => type == "uint" || type == "min16uint" ? "0u" : type == "int" || type == "min16int" ? "0" : type == "bool" ? "false" : "0.0f";
    private static string Zero(Field field) => field.Width == 1 ? ScalarZero(field.BaseType) : field.Type + "(" + string.Join(", ", Enumerable.Repeat(ScalarZero(field.BaseType), field.Width)) + ")";
    private static string SemanticStem(string semantic) => Regex.Replace(semantic, @"\d+$", "");
    private static ulong SemanticIndex(string semantic)
    {
        string suffix = Regex.Match(semantic, @"\d+$").Value;
        if (suffix.Length == 0) return 0;
        if (!ulong.TryParse(suffix, out ulong value)) throw new Unsupported("UNSUPPORTED_SEMANTIC_INDEX");
        return value;
    }
    private static string InterpolationMode(string modifiers)
    {
        var values = modifiers.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToHashSet(StringComparer.Ordinal);
        if (values.Count != modifiers.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length
            || (values.Contains("nointerpolation") && values.Count != 1)
            || (values.Contains("linear") && values.Contains("noperspective"))
            || (values.Contains("sample") && values.Contains("centroid")))
            throw new Unsupported("AMBIGUOUS_INTERPOLATION_MODIFIERS");
        values.Remove("linear"); // implicit default and explicit linear are equivalent.
        return string.Join(" ", values.OrderBy(value => value, StringComparer.Ordinal));
    }
    private static bool IsRasterizerGenerated(string semantic) => semantic.Equals("SV_IsFrontFace", StringComparison.OrdinalIgnoreCase)
        || semantic.Equals("SV_PrimitiveID", StringComparison.OrdinalIgnoreCase) || semantic.Equals("SV_SampleIndex", StringComparison.OrdinalIgnoreCase)
        || semantic.Equals("SV_Coverage", StringComparison.OrdinalIgnoreCase);
    private static void ValidateSystem(Field field, bool vertex)
    {
        if (!field.System) return;
        bool valid = field.Key switch
        {
            "SV_POSITION" => field.BaseType == "float" && field.Width == 4,
            "SV_RENDERTARGETARRAYINDEX" or "SV_VIEWPORTARRAYINDEX" => field.BaseType == "uint" && field.Width == 1,
            "SV_ISFRONTFACE" => !vertex && field.BaseType == "bool" && field.Width == 1,
            "SV_PRIMITIVEID" or "SV_SAMPLEINDEX" or "SV_COVERAGE" => !vertex && field.BaseType == "uint" && field.Width == 1,
            _ => false,
        };
        if (!valid) throw new Unsupported("UNSUPPORTED_SYSTEM_SEMANTIC: " + field.Semantic);
    }
    private static int MatchingBrace(string source, int open)
    {
        int depth = 0;
        for (int index = open; index < source.Length; index++)
        {
            if (source[index] == '{') depth++;
            else if (source[index] == '}' && --depth == 0) return index;
        }
        throw new Unsupported("UNBALANCED_SOURCE_BRACES");
    }
    private static string MaskTrivia(string source)
    {
        char[] mask = source.ToCharArray();
        for (int index = 0; index < source.Length; index++)
        {
            int start = index, end = index;
            if (source[index] == '/' && index + 1 < source.Length && source[index + 1] == '/')
            {
                end = source.IndexOf('\n', index + 2);
                if (end < 0) end = source.Length;
            }
            else if (source[index] == '/' && index + 1 < source.Length && source[index + 1] == '*')
            {
                end = source.IndexOf("*/", index + 2, StringComparison.Ordinal);
                if (end < 0) throw new Unsupported("UNTERMINATED_SOURCE_COMMENT");
                end += 2;
            }
            else if (source[index] == '"' || source[index] == '\'')
            {
                char quote = source[index];
                end = index + 1;
                while (end < source.Length && source[end] != quote)
                {
                    if (source[end] == '\\') end++;
                    end++;
                }
                if (end >= source.Length) throw new Unsupported("UNTERMINATED_SOURCE_LITERAL");
                end++;
            }
            if (end <= start) continue;
            for (int cursor = start; cursor < end; cursor++) if (mask[cursor] != '\r' && mask[cursor] != '\n') mask[cursor] = ' ';
            index = end - 1;
        }
        return new string(mask);
    }
}
