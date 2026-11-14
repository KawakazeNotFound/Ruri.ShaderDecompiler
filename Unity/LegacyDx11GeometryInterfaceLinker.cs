using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace Ruri.ShaderTools;

public sealed class LegacyDx11GeometryInterfaceLinkResult
{
    public bool Success { get; internal set; }
    public string Status { get; internal set; } = "PARTIAL";
    public string Diagnostic { get; internal set; } = "";
    public string Evidence { get; internal set; } = "SOURCE_INTERFACE_ONLY; DXBC_FXC_UNITY_NOT_RUN";
    public int CompatibleTriples { get; internal set; }
    public int AddedPaddingFields { get; internal set; }
    public int ReorderedInterfaces { get; internal set; }
    public int PropagatedFlatFields { get; internal set; }
    public List<LegacyDx11GeometrySourceProof> SourceProofs { get; } = new();
}

public sealed class LegacyDx11GeometrySourceProof
{
    public string Stage { get; internal set; } = "";
    public uint BlobIndex { get; internal set; }
    public string BeforeSha256 { get; internal set; } = "";
    public string AfterSha256 { get; internal set; } = "";
    public List<string> PaddingSemantics { get; } = new();
}

/// <summary>
/// Restricted VS -> triangle GS -> PS source-interface linker. It changes only
/// struct declarations, transactionally. Never edits main, primitive arity,
/// maxvertexcount, stream operations, storage, or arithmetic. The source proof
/// deliberately does not claim original DXBC identity, compilation, or rendering.
/// </summary>
public static class LegacyDx11GeometryInterfaceLinker
{
    private sealed class Unsupported : Exception { public Unsupported(string message) : base(message) { } }
    private sealed record Field(string Name, string Semantic, string Type, string Mode)
    {
        public bool System => Semantic.StartsWith("SV_", StringComparison.OrdinalIgnoreCase);
        public string Key => Regex.Replace(Semantic.ToUpperInvariant(), @"\d+$", "") + Index.ToString(CultureInfo.InvariantCulture);
        private ulong Index => ulong.TryParse(Regex.Match(Semantic, @"\d+$").Value, out ulong n) ? n : 0;
        public string Declaration => (Mode.Length == 0 ? "" : Mode + " ") + Type + " " + Name + " : " + Semantic + ";";
    }
    private sealed class Interface
    {
        public int Start;
        public int Length;
        public List<Field> Fields = new();
    }
    private sealed class Model
    {
        public UnitySerializedSubProgram Program = null!;
        public string Stage = "";
        public string Source = "";
        public string Mask = "";
        public Interface Input = null!;
        public Interface Output = null!;
        public List<(Interface Region, List<Field> Fields)> Plans = new();
        public List<string> Padding = new();
        public HashSet<ushort> Keywords = new();
    }
    private static readonly Regex FieldPattern = new(@"^\s*(?<mode>(?:(?:linear|centroid|noperspective|sample|nointerpolation)\s+)*)(?<type>(?:float|half|min16float|min10float|int|uint|min16int|min16uint|bool)[1-4]?)\s+(?<name>[A-Za-z_]\w*)\s*:\s*(?<sem>[A-Za-z_]\w*)\s*;\s*$", RegexOptions.CultureInvariant);

    public static LegacyDx11GeometryInterfaceLinkResult Link(UnitySerializedPass pass, bool allowPadding = true)
    {
        ArgumentNullException.ThrowIfNull(pass);
        var result = new LegacyDx11GeometryInterfaceLinkResult();
        try
        {
            var slots = pass.EnumerateProgramSlots().Where(x => x.Program.SubPrograms.Count > 0).ToList();
            if (slots.Count != 3 || slots.Any(x => x.Stage != "Vertex" && x.Stage != "Geometry" && x.Stage != "Fragment"))
                throw new Unsupported("UNSUPPORTED_STAGE_TOPOLOGY: expected exactly Vertex/Geometry/Fragment");
            var vertices = pass.ProgVertex!.SubPrograms.Select(x => Parse(x, "Vertex")).ToList();
            var geometries = pass.ProgGeometry!.SubPrograms.Select(x => Parse(x, "Geometry")).ToList();
            var pixels = pass.ProgFragment!.SubPrograms.Select(x => Parse(x, "Fragment")).ToList();
            ValidateVariants(vertices); ValidateVariants(geometries); ValidateVariants(pixels);
            var vg = Shared(vertices, geometries);
            var gp = Shared(geometries, pixels);
            var vp = Shared(vertices, pixels);
            var triples = (from v in vertices from g in geometries from p in pixels
                           where Compatible(v, g, vg) && Compatible(g, p, gp) && Compatible(v, p, vp)
                           select (V: v, G: g, P: p)).ToList();
            if (triples.Count == 0 || vertices.Any(x => !triples.Any(t => t.V == x)) || geometries.Any(x => !triples.Any(t => t.G == x)) || pixels.Any(x => !triples.Any(t => t.P == x)))
                throw new Unsupported("UNPAIRED_VARIANT: all three stages require compatible-keyword partners");
            result.CompatibleTriples = triples.Count;
            // Only PS-explicit flat may propagate, and only if EVERY compatible
            // consumed PS requires it with exactly the existing dtype/width.
            foreach (Model geometry in geometries)
            {
                var consumers = triples.Where(t => t.G == geometry).Select(t => t.P).Distinct().ToList();
                foreach (Field output in geometry.Output.Fields.ToArray())
                {
                    var uses = consumers.Select(p => p.Input.Fields.SingleOrDefault(f => f.Key == output.Key)).Where(f => f != null).Cast<Field>().ToList();
                    if (!output.System && uses.Count != 0 && output.Mode.Length == 0 && uses.All(f => f.Type == output.Type && f.Mode == "nointerpolation"))
                    {
                        if (!allowPadding) throw new Unsupported("QUALIFIER_REQUIRES_PROPAGATION: " + output.Key);
                        ReplacePlannedField(geometry, geometry.Output, output with { Mode = "nointerpolation" });
                        result.PropagatedFlatFields++;
                    }
                }
            }
            foreach (Model geometry in geometries)
                PlanConsumer(geometry, geometry.Input, triples.Where(t => t.G == geometry).Select(t => t.V).Distinct().ToList(), false, allowPadding, result);
            foreach (Model pixel in pixels)
                PlanConsumer(pixel, pixel.Input, triples.Where(t => t.P == pixel).Select(t => t.G).Distinct().ToList(), true, allowPadding, result);
            var all = vertices.Concat(geometries).Concat(pixels).ToList();
            var revised = new Dictionary<Model, string>();
            foreach (Model model in all)
            {
                string source = model.Source;
                foreach (var plan in model.Plans.OrderByDescending(x => x.Region.Start))
                {
                    string newline = source.Contains("\r\n", StringComparison.Ordinal) ? "\r\n" : "\n";
                    string body = newline + String.Join(newline, plan.Fields.Select(f => "    " + f.Declaration)) + newline;
                    source = source.Remove(plan.Region.Start, plan.Region.Length).Insert(plan.Region.Start, body);
                }
                revised.Add(model, source);
            }
            // Commit only after every variant family and both edges passed.
            foreach (Model model in all)
            {
                string source = revised[model];
                var proof = new LegacyDx11GeometrySourceProof { Stage = model.Stage, BlobIndex = model.Program.BlobIndex, BeforeSha256 = Hash(model.Source), AfterSha256 = Hash(source) };
                proof.PaddingSemantics.AddRange(model.Padding);
                result.SourceProofs.Add(proof);
                model.Program.SourceCode = source;
            }
            result.Success = true;
            result.Status = all.Any(m => revised[m] != m.Source) ? "SOURCE_INTERFACE_LINKED" : "EXACT_SOURCE_CHAIN_VALIDATED";
        }
        catch (Unsupported ex)
        {
            result.Diagnostic = ex.Message;
            // Planned work is not committed evidence. Failure leaves all sources
            // unchanged and every mutation/proof counter empty.
            result.CompatibleTriples = 0; result.AddedPaddingFields = 0;
            result.ReorderedInterfaces = 0; result.PropagatedFlatFields = 0;
            result.SourceProofs.Clear();
        }
        return result;
    }

    private static void PlanConsumer(Model consumer, Interface input, List<Model> producers, bool rasterEdge, bool allowPadding, LegacyDx11GeometryInterfaceLinkResult result)
    {
        var existing = Fields(consumer, input);
        var rasterInputs = rasterEdge ? existing.Where(IsRasterizerGenerated).ToList() : new List<Field>();
        var consumedInputs = existing.Except(rasterInputs).ToList();
        // Extra system outputs are not padding: SV_Position goes to rasterization,
        // and VS-only array-index outputs do not become synthetic GS inputs.
        var candidates = producers.Select(p =>
        {
            var fields = Fields(p, p.Output);
            if (fields.Any(f => f.System && !consumedInputs.Any(e => e.Key == f.Key)
                && !f.Semantic.Equals("SV_Position", StringComparison.OrdinalIgnoreCase)
                && !f.Semantic.Equals("SV_RenderTargetArrayIndex", StringComparison.OrdinalIgnoreCase)))
                throw new Unsupported("UNSUPPORTED_UNCONSUMED_SYSTEM_SEMANTIC");
            fields = fields.Where(f => !f.System || consumedInputs.Any(e => e.Key == f.Key)).ToList();
            // Producer-only suffixes do not precede any consumed field, and thus
            // need no synthetic consumer declaration. A middle slot is retained.
            int lastConsumed = fields.FindLastIndex(f => consumedInputs.Any(e => e.Key == f.Key));
            return fields.Take(lastConsumed + 1).ToList();
        }).ToList();
        string Layout(List<Field> fields) => String.Join("|", fields.Select(f => f.Key + ":" + f.Type + ":" + f.Mode));
        if (candidates.Select(Layout).Distinct(StringComparer.Ordinal).Count() != 1)
            throw new Unsupported("VARIANT_FAMILY_LAYOUT_CONFLICT: " + consumer.Stage + " blob " + consumer.Program.BlobIndex);
        List<Field> producer = candidates[0];
        var desired = new List<Field>();
        var names = existing.Select(x => x.Name).ToHashSet(StringComparer.Ordinal);
        foreach (Field field in producer)
        {
            Field? target = consumedInputs.SingleOrDefault(x => x.Key == field.Key);
            if (target == null)
            {
                if (field.System) throw new Unsupported("UNSUPPORTED_SYSTEM_PADDING: " + field.Key);
                if (!allowPadding) throw new Unsupported("PRODUCER_SLOT_ABSENT_FROM_CONSUMER: " + field.Key);
                string name = "_ruri_geometry_padding_" + field.Key;
                if (!names.Add(name) || Regex.IsMatch(consumer.Mask, @"\b" + Regex.Escape(name) + @"\b"))
                    throw new Unsupported("PADDING_NAME_COLLISION");
                target = field with { Name = name };
                desired.Add(target);
                consumer.Padding.Add(field.Key);
                result.AddedPaddingFields++;
                continue;
            }
            if (target.Type != field.Type) throw new Unsupported("INCOMPATIBLE_FIELD_TYPE: " + field.Key + " " + field.Type + "->" + target.Type);
            if (rasterEdge && NormalizeMode(target.Mode) != NormalizeMode(field.Mode))
                throw new Unsupported("INCOMPATIBLE_INTERPOLATION: " + field.Key);
            desired.Add(target);
        }
        desired.AddRange(rasterInputs);
        if (existing.Any(x => !desired.Any(d => d.Key == x.Key))) throw new Unsupported("CONSUMER_FIELD_HAS_NO_PRODUCER");
        if (existing.Select(x => x.Key).SequenceEqual(desired.Select(x => x.Key))) return;
        if (!allowPadding) throw new Unsupported("INTERFACE_ORDER_MISMATCH");
        if (existing.Count == desired.Count) result.ReorderedInterfaces++;
        SetPlan(consumer, input, desired);
    }
    private static List<Field> Fields(Model model, Interface region) => model.Plans.FirstOrDefault(x => x.Region == region).Fields ?? region.Fields;
    private static void ReplacePlannedField(Model model, Interface region, Field field)
        => SetPlan(model, region, Fields(model, region).Select(x => x.Key == field.Key ? field : x).ToList());
    private static void SetPlan(Model model, Interface region, List<Field> fields)
    {
        model.Plans.RemoveAll(x => x.Region == region);
        model.Plans.Add((region, fields));
    }
    private static string NormalizeMode(string mode) => mode.Length == 0 || mode == "linear" ? "linear" : mode;
    private static bool IsRasterizerGenerated(Field field) =>
        (field.Semantic.Equals("SV_IsFrontFace", StringComparison.OrdinalIgnoreCase) && field.Type == "bool") ||
        (field.Semantic.Equals("SV_SampleIndex", StringComparison.OrdinalIgnoreCase) && field.Type == "uint") ||
        (field.Semantic.Equals("SV_Coverage", StringComparison.OrdinalIgnoreCase) && field.Type == "uint") ||
        (field.Semantic.Equals("SV_Position", StringComparison.OrdinalIgnoreCase) && field.Type == "float4");
    private static void ValidateVariants(List<Model> models)
    {
        var keys = new HashSet<string>(StringComparer.Ordinal);
        foreach (Model model in models)
            if (!keys.Add(String.Join(",", model.Keywords.OrderBy(k => k))))
                throw new Unsupported("DUPLICATE_VARIANT_CONDITION: source selection is ambiguous");
    }
    private static HashSet<ushort> Shared(List<Model> first, List<Model> second)
    {
        var set = first.SelectMany(x => x.Keywords).ToHashSet(); set.IntersectWith(second.SelectMany(x => x.Keywords)); return set;
    }
    private static bool Compatible(Model a, Model b, HashSet<ushort> shared) => shared.All(k => a.Keywords.Contains(k) == b.Keywords.Contains(k));
    private static Model Parse(UnitySerializedSubProgram program, string stage)
    {
        if (!program.Success || program.SourceLanguage != "hlsl" || String.IsNullOrWhiteSpace(program.SourceCode)) throw new Unsupported("MISSING_HLSL_SOURCE");
        if (program.KeywordIndices.Count != program.KeywordIndices.Distinct().Count()) throw new Unsupported("DUPLICATE_KEYWORD_INDEX");
        var model = new Model { Program = program, Stage = stage, Source = program.SourceCode!, Mask = Mask(program.SourceCode!), Keywords = program.KeywordIndices.ToHashSet() };
        model.Input = ParseInterface(model.Mask, "SPIRV_Cross_Input");
        model.Output = ParseInterface(model.Mask, "SPIRV_Cross_Output");
        ValidateInterfaceUses(model.Mask);
        foreach (Field field in model.Input.Fields.Where(f => f.System)) ValidateSystem(field, stage, true);
        foreach (Field field in model.Output.Fields.Where(f => f.System)) ValidateSystem(field, stage, false);
        if (stage == "Geometry")
        {
            const string entry = @"\[maxvertexcount\((?<max>\d+)\)\]\s*void\s+main\(\s*triangle\s+SPIRV_Cross_Input\s+stage_input\[3\]\s*,\s*inout\s+TriangleStream<SPIRV_Cross_Output>\s+geometry_stream\s*\)\s*\{";
            var matches = Regex.Matches(model.Mask, entry);
            if (matches.Count != 1 || !Int32.TryParse(matches[0].Groups["max"].Value, out int max) || max < 1 || max > 1024)
                throw new Unsupported("UNSUPPORTED_GEOMETRY_ENTRY: expected one triangle[3]/TriangleStream/maxvertexcount");
            if (!model.Output.Fields.Any(f => f.Semantic.Equals("SV_Position", StringComparison.OrdinalIgnoreCase) && f.Type == "float4"))
                throw new Unsupported("MISSING_GEOMETRY_SV_POSITION");
            if (Regex.Matches(model.Mask, @"\bTriangleStream\s*<\s*(?<type>[A-Za-z_]\w*)\s*>").Cast<Match>().Any(m => m.Groups["type"].Value != "SPIRV_Cross_Output")
                || Regex.IsMatch(model.Mask, @"\b(?:PointStream|LineStream)\s*<|\[\s*instance\s*\("))
                throw new Unsupported("UNSUPPORTED_GEOMETRY_STREAM_TOPOLOGY");
        }
        else if (Regex.Matches(model.Mask, @"\bSPIRV_Cross_Output\s+main\(\s*SPIRV_Cross_Input\s+stage_input\s*\)\s*\{").Count != 1)
            throw new Unsupported("UNSUPPORTED_ENTRY_SHAPE: " + stage);
        return model;
    }
    private static void ValidateSystem(Field field, string stage, bool input)
    {
        string semantic = Regex.Replace(field.Semantic.ToUpperInvariant(), @"\d+$", "");
        bool floats = Regex.IsMatch(field.Type, @"\Afloat[1-4]?\z");
        bool allowed = (stage, input, semantic) switch
        {
            ("Vertex", true, "SV_VERTEXID" or "SV_INSTANCEID") => field.Type == "uint",
            ("Fragment", true, "SV_ISFRONTFACE") => field.Type == "bool",
            ("Fragment", true, "SV_SAMPLEINDEX" or "SV_COVERAGE") => field.Type == "uint",
            ("Fragment", false, "SV_TARGET") => floats,
            ("Fragment", false, "SV_DEPTH" or "SV_DEPTHGREATEREQUAL" or "SV_DEPTHLESSEQUAL") => field.Type == "float",
            (_, _, "SV_POSITION") => field.Type == "float4" && !(stage == "Fragment" && !input),
            (_, _, "SV_RENDERTARGETARRAYINDEX" or "SV_VIEWPORTARRAYINDEX") => field.Type == "uint" && !(stage == "Vertex" && input),
            ("Geometry", false, "SV_PRIMITIVEID") or ("Fragment", true, "SV_PRIMITIVEID") => field.Type == "uint",
            (_, _, "SV_CLIPDISTANCE" or "SV_CULLDISTANCE") => floats && !(stage == "Vertex" && input) && !(stage == "Fragment" && !input),
            _ => false,
        };
        if (!allowed) throw new Unsupported("UNSUPPORTED_SYSTEM_SEMANTIC_OR_TYPE: " + stage + ":" + (input ? "input:" : "output:") + field.Semantic);
    }
    private static void ValidateInterfaceUses(string source)
    {
        // Struct layout edits are sound only for the decompiler's named-member
        // subset. Positional initializers/casts and buffer-backed interface
        // layouts need an IR proof; their source text staying equal is insufficient.
        const string type = @"SPIRV_Cross_(?:Input|Output)";
        string body = Regex.Replace(source, @"\bstruct\s+" + type + @"\s*\{[^{}]*\}\s*;", m => new string(m.Value.Select(c => c == '\r' || c == '\n' ? c : ' ').ToArray()));
        // Alias resolution is outside the restricted named-member proof. Reject
        // aliases before their later aggregate/buffer use can hide the type.
        if (Regex.IsMatch(body, @"\btypedef\b[^;{}]*\b" + type + @"\b"))
            throw new Unsupported("UNSUPPORTED_INTERFACE_TYPEDEF_ALIAS");
        if (Regex.IsMatch(body, @"\(\s*" + type + @"\s*\)|\b" + type + @"\s*\(|\b" + type + @"\s+[A-Za-z_]\w*\s*\{"))
            throw new Unsupported("UNSUPPORTED_INTERFACE_POSITIONAL_CAST_OR_INITIALIZER");
        foreach (Match declaration in Regex.Matches(body, @"\b" + type + @"\s+(?<name>[A-Za-z_]\w*)(?:\s*\[[^\]\r\n]+\])?"))
        {
            string name = Regex.Escape(declaration.Groups["name"].Value);
            if (Regex.IsMatch(body, @"\b" + name + @"(?:\s*\[[^\]\r\n]+\])?\s*=\s*\{"))
                throw new Unsupported("UNSUPPORTED_INTERFACE_POSITIONAL_INITIALIZER");
            if (Regex.IsMatch(body, @"\b(?:asfloat|asuint|asint)\s*\(\s*" + name + @"(?:\s*\[[^\]\r\n]+\])?\s*[,)]"))
                throw new Unsupported("UNSUPPORTED_INTERFACE_BITCAST");
        }
        foreach (Match generic in Regex.Matches(body, @"\b(?<container>[A-Za-z_]\w*)\s*<\s*SPIRV_Cross_(?<kind>Input|Output)\s*>"))
            if (generic.Groups["container"].Value != "TriangleStream" || generic.Groups["kind"].Value != "Output")
                throw new Unsupported("UNSUPPORTED_BUFFER_BACKED_INTERFACE_LAYOUT");
    }
    private static Interface ParseInterface(string source, string name)
    {
        var matches = Regex.Matches(source, @"\bstruct\s+" + name + @"\s*\{(?<body>[^{}]*)\}\s*;");
        if (matches.Count != 1) throw new Unsupported("UNSUPPORTED_INTERFACE_STRUCT: " + name);
        Group body = matches[0].Groups["body"];
        var region = new Interface { Start = body.Index, Length = body.Length };
        var keys = new HashSet<string>(StringComparer.Ordinal);
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (string line in body.Value.Replace("\r", "").Split('\n'))
        {
            if (String.IsNullOrWhiteSpace(line)) continue;
            Match match = FieldPattern.Match(line);
            if (!match.Success) throw new Unsupported("UNSUPPORTED_INTERFACE_FIELD");
            var field = new Field(match.Groups["name"].Value, match.Groups["sem"].Value, match.Groups["type"].Value, Regex.Replace(match.Groups["mode"].Value.Trim(), @"\s+", " "));
            string digits = Regex.Match(field.Semantic, @"\d+$").Value;
            if (digits.Length != 0 && !UInt64.TryParse(digits, NumberStyles.None, CultureInfo.InvariantCulture, out _))
                throw new Unsupported("UNSUPPORTED_SEMANTIC_INDEX");
            if (!keys.Add(field.Key)) throw new Unsupported("DUPLICATE_SEMANTIC: " + field.Key);
            if (!names.Add(field.Name)) throw new Unsupported("DUPLICATE_INTERFACE_FIELD_NAME");
            var modes = field.Mode.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (modes.Length != modes.Distinct().Count() || modes.Count(m => m is "linear" or "noperspective" or "nointerpolation") > 1
                || (modes.Contains("centroid") && modes.Contains("sample"))
                || (modes.Contains("nointerpolation") && modes.Length != 1))
                throw new Unsupported("CONTRADICTORY_INTERPOLATION_MODIFIERS");
            region.Fields.Add(field);
        }
        return region;
    }
    private static string Hash(string source) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(source))).ToLowerInvariant();
    private static string Mask(string source)
    {
        // Preserve offsets/newlines while excluding comments and string literals.
        return Regex.Replace(source, @"//[^\r\n]*|/\*[\s\S]*?\*/|""(?:\\.|[^""\\])*""", m => new string(m.Value.Select(c => c == '\r' || c == '\n' ? c : ' ').ToArray()));
    }
}
