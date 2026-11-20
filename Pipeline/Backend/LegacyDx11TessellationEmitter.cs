using System.Globalization;
using System.Numerics;
using System.Text;
using Ruri.ShaderTools.Pipeline.Frontend;
using Silk.NET.SPIRV;
using Variable = Ruri.ShaderTools.Pipeline.Backend.LegacyDx11TessellationModule.Variable;
using static Ruri.ShaderTools.Pipeline.Backend.LegacyDx11TessellationModule;

namespace Ruri.ShaderTools.Pipeline.Backend;

internal static class LegacyDx11TessellationEmitter
{
    private const uint Position = 1;
    private const uint QuadEdge = 11;
    private const uint QuadInside = 12;
    private const uint TriangleEdge = 13;
    private const uint TriangleInside = 14;
    private const uint LineDetail = 15;
    private const uint LineDensity = 16;
    private const string Input = "RuriTessellationInput";
    private const string Output = "RuriTessellationOutput";
    private const string PatchData = "RuriTessellationPatchData";
    private const string PatchEntry = "RuriTessellationPatchEntry";

    private sealed record Field(string Name, string Semantic, string Type, int Count, Variable? Variable);

    public static string Emit(ReadOnlySpan<byte> dxbc, byte[] spirv, EmissionPlan plan, SpirvCrossDriver driver)
    {
        DxbcTessellation declarations = DxbcTessellation.Read(dxbc);
        bool isHull = declarations.Stage == PipelineStage.TessControl;
        if (declarations.Stage != plan.EntryPoint.Stage) throw Unsupported("DXBC and SPIR-V stages disagree");

        var lowered = Lower(spirv, declarations);
        var inputs = DxbcSignature.Read(dxbc, DxbcSignatureKind.Input);
        var outputs = DxbcSignature.Read(dxbc, DxbcSignatureKind.Output);
        var constants = DxbcSignature.Read(dxbc, DxbcSignatureKind.PatchConstant);
        var used = new HashSet<uint>();
        int GetPatchOffset() => checked((int)lowered.Variables.Where(v => v.Input != isHull && !v.Patch && v.Location.HasValue)
            .Select(v => v.Location!.Value / 4 + 1).DefaultIfEmpty().Max());
        Field[] inputFields = ReadFields(inputs, lowered.Variables, used, input: true, isPatch: false, 0);
        Field[] outputFields = ReadFields(outputs, lowered.Variables, used, input: false, isPatch: false, 0);
        Field[] patchFields = ReadPatchFields(declarations.Domain, constants, lowered.Variables, used, isHull, GetPatchOffset());
        if (outputFields.Length == 0) throw Unsupported("missing output signature");

        Variable? invocation = ReadBuiltIn(BuiltIn.InvocationId, "uint", 1);
        Variable? primitive = ReadBuiltIn(BuiltIn.PrimitiveId, "uint", 1);
        Variable? inputCount = ReadBuiltIn(BuiltIn.PatchVertices, "uint", 1);
        Variable? location = ReadBuiltIn(BuiltIn.TessCoord, "float", 3);
        if (isHull ? location is not null : invocation is not null)
            throw Unsupported("unexpected hull/domain system value");
        if (lowered.Variables.Any(v => !used.Contains(v.Id)))
            throw Unsupported("SPIR-V input or output does not map to the DXBC signature");

        foreach (Variable var in inputFields.Where(f => f.Variable != null).Select(f => f.Variable!))
            if (var.Count < declarations.InputControlPoints) throw Unsupported("insufficient SPIR-V input control points");
        foreach (Variable var in outputFields.Select(f => f.Variable!))
            if (var.Count != (isHull ? (uint)declarations.OutputControlPoints : 0))
                throw Unsupported("SPIR-V output control point size differs from DXBC declaration");

        var carrier = new EmissionPlan { ShaderModel = plan.ShaderModel, Language = EmitLanguage.Hlsl, EntryPoint = lowered.Entry };
        carrier.FlattenedBlocks.AddRange(plan.FlattenedBlocks);
        string source = driver.Emit(lowered.Bytes, carrier) ?? throw Unsupported(driver.LastFailure ?? "SPIRV-Cross emitted no HLSL body");
        int entry = source.LastIndexOf("\nvoid vert_main()", StringComparison.Ordinal);
        if (entry == -1 || !source.AsSpan(entry).Contains("\nvoid main()", StringComparison.Ordinal) ||
            (isHull ? !HasFunction(Control) || !HasFunction(Patch) : !HasFunction(Evaluate)))
            throw Unsupported("unexpected SPIRV-Cross carrier entry or tessellation body spelling");

        bool HasFunction(string name) => source.Contains("void " + name + "()", StringComparison.Ordinal);
        var text = new StringBuilder(source.AsSpan(0, entry).TrimEnd().ToString());
        text.AppendLine().AppendLine();
        if (inputFields.Length != 0) WriteStruct(text, Input, inputFields);
        WriteStruct(text, Output, outputFields);
        WriteStruct(text, PatchData, patchFields);

        string domain = declarations.Domain switch
        {
            DxbcTessellationDomain.Isoline => "isoline",
            DxbcTessellationDomain.Triangle => "tri",
            DxbcTessellationDomain.Quad => "quad",
            _ => throw Unsupported("unknown tessellation domain")
        };
        string primitiveId = primitive is null ? "" : "uint primitiveId : SV_PrimitiveID";
        string inputPatch = inputFields.Length == 0 ? "" :
            $"{(isHull ? "InputPatch" : "OutputPatch")}<{Input}, {declarations.InputControlPoints}> inputPatch";

        if (isHull)
        {
            string outputPatch = $"OutputPatch<{Output}, {declarations.OutputControlPoints}> outputPatch";
            text.AppendLine($"{PatchData} {PatchEntry}({Parameters(inputPatch, outputPatch, primitiveId)})").AppendLine("{");
            CopyInputs();
            CopyControlPoints(outputFields, declarations.OutputControlPoints, "outputPatch");
            if (invocation is not null) text.AppendLine($"    {invocation.Name} = 0u;");
            text.AppendLine($"    {Patch}();");
            Return(PatchData, patchFields, "");
            text.AppendLine("}").AppendLine();

            string partition = declarations.Partition switch
            {
                DxbcTessellationPartition.Integer => "integer",
                DxbcTessellationPartition.PowerOfTwo => "pow2",
                DxbcTessellationPartition.FractionalOdd => "fractional_odd",
                DxbcTessellationPartition.FractionalEven => "fractional_even",
                _ => throw Unsupported("unknown tessellation partition")
            };
            string topology = declarations.Topology switch
            {
                DxbcTessellationTopology.Point => "point",
                DxbcTessellationTopology.Line => "line",
                DxbcTessellationTopology.TriangleClockwise => "triangle_cw",
                DxbcTessellationTopology.TriangleCounterClockwise => "triangle_ccw",
                _ => throw Unsupported("unknown tessellation topology")
            };
            text.AppendLine($"[domain(\"{domain}\")]").AppendLine($"[partitioning(\"{partition}\")]")
                .AppendLine($"[outputtopology(\"{topology}\")]").AppendLine($"[outputcontrolpoints({declarations.OutputControlPoints})]")
                .AppendLine($"[patchconstantfunc(\"{PatchEntry}\")]");
            if (declarations.MaximumFactor is float limit)
            {
                string value = limit.ToString("R", CultureInfo.InvariantCulture);
                text.AppendLine("[maxtessfactor(" + (value.Contains('.') ? value : value + ".0") + ")]");
            }
            text.AppendLine($"{Output} main({Parameters(inputPatch, "uint controlPoint : SV_OutputControlPointID", primitiveId)})").AppendLine("{");
            CopyInputs();
            if (invocation is not null) text.AppendLine($"    {invocation.Name} = controlPoint;");
            text.AppendLine($"    {Control}();");
            Return(Output, outputFields, "[controlPoint]");
        }
        else
        {
            bool triangle = declarations.Domain == DxbcTessellationDomain.Triangle;
            string tessLocation = $"float{(triangle ? 3 : 2)} tessLocation : SV_DomainLocation";
            text.AppendLine($"[domain(\"{domain}\")]");
            text.AppendLine($"{Output} main({Parameters($"{PatchData} patchData", tessLocation, inputPatch, primitiveId)})").AppendLine("{");
            CopyInputs();
            if (location is not null)
                text.AppendLine($"    {location.Name} = {(triangle ? "tessLocation" : "float3(tessLocation, 0.0f)")};");
            foreach (Field field in patchFields.Where(f => f.Variable is not null))
            {
                string swizzle = ReadSwizzle(field);
                if (field.Count == 0) text.AppendLine($"    {field.Variable!.Name} = patchData.{field.Name}{swizzle};");
                for (int index = 0; index < field.Count; index++)
                    text.AppendLine($"    {field.Variable!.Name}[{index}] = patchData.{field.Name}[{index}];");
            }
            text.AppendLine($"    {Evaluate}();");
            Return(Output, outputFields, "");
        }
        text.AppendLine("}");
        return text.ToString();

        Variable? ReadBuiltIn(BuiltIn builtin, string type, uint width)
        {
            Variable? var = lowered.Variables.SingleOrDefault(v => v.BuiltIn == builtin);
            if (var is not null && (!var.Input || var.Patch || var.Count != 0 || var.Type != type || var.Width != width || !used.Add(var.Id)))
                throw Unsupported($"unexpected type or direction of {builtin}");
            return var;
        }

        void CopyInputs()
        {
            CopyControlPoints(inputFields, declarations.InputControlPoints, "inputPatch");
            if (inputCount is not null) text.AppendLine($"    {inputCount.Name} = {declarations.InputControlPoints}u;");
            if (primitive is not null) text.AppendLine($"    {primitive.Name} = primitiveId;");
        }

        void CopyControlPoints(Field[] fields, int count, string variable)
        {
            foreach (Field field in fields.Where(f => f.Variable is not null))
            {
                string swizzle = ReadSwizzle(field);
                for (int index = 0; index < count; index++)
                    text.AppendLine($"    {field.Variable!.Name}[{index}] = {variable}[{index}].{field.Name}{swizzle};");
            }
        }

        void Return(string type, Field[] fields, string index)
        {
            text.AppendLine($"    {type} result;");
            foreach (Field field in fields)
            {
                if (field.Count == 0) text.AppendLine($"    result.{field.Name} = {field.Variable!.Name}{index};");
                else for (int item = 0; item < field.Count; item++)
                    text.AppendLine($"    result.{field.Name}[{item}] = {field.Variable!.Name}[{item}];");
            }
            text.AppendLine("    return result;");
        }
    }

    private static Field[] ReadFields(IEnumerable<DxbcSignatureElement> elements, IReadOnlyList<Variable> variables,
        HashSet<uint> used, bool input, bool isPatch, int locationOffset)
    {
        var fields = new List<Field>();
        foreach (var signature in elements)
        {
            if (signature.Mask is < 1 or > 15 || signature.Stream != 0 || signature.MinPrecision != 0 || signature.Register >= 32 ||
                !signature.SemanticName.All(c => char.IsAsciiLetterOrDigit(c) || c == '_') || !char.IsAsciiLetter(signature.SemanticName[0]))
                throw Unsupported($"unsupported signature element {signature.SemanticName}{signature.SemanticIndex}");
            string scalar = signature.ComponentType switch
            {
                1 => "uint", 2 => "int", 3 => "float", _ => throw Unsupported("unknown DXBC signature component type")
            };
            int width = BitOperations.PopCount(signature.Mask);
            int lane = BitOperations.TrailingZeroCount(signature.Mask);
            if (((1 << width) - 1) << lane != signature.Mask)
                throw Unsupported("nonconsecutive lanes in a DXBC tessellation signature");

            uint location = checked((signature.Register + (uint)locationOffset) * 4 + (uint)lane);
            string rawName = signature.SemanticName + (signature.SemanticIndex != 0 ? "_" + signature.SemanticIndex.ToString(CultureInfo.InvariantCulture) : "");
            Variable? variable = variables.SingleOrDefault(v => v.Input == input && v.Patch == isPatch &&
                (v.BuiltIn.HasValue || string.Equals(v.OriginalName, rawName, StringComparison.OrdinalIgnoreCase)) &&
                (v.Location == location || (signature.SystemValue == Position && v.BuiltIn == BuiltIn.Position)));
            if (variable is not null && (variable.Type != scalar || variable.Width > width || (!input && variable.Width != width) || !used.Add(variable.Id)))
                throw Unsupported($"DXBC and SPIR-V types disagree for {signature.SemanticName}{signature.SemanticIndex}");
            if (variable is null && (!input || (signature.UsedMask & signature.Mask) != 0))
                throw Unsupported($"missing SPIR-V variable for {signature.SemanticName}{signature.SemanticIndex}");
            string semantic = signature.SemanticName + (signature.SemanticIndex != 0 ? signature.SemanticIndex.ToString(CultureInfo.InvariantCulture) : "");
            fields.Add(new("Field" + fields.Count, semantic, Type(scalar, width), 0, variable));
        }
        return fields.ToArray();
    }

    private static Field[] ReadPatchFields(DxbcTessellationDomain domain, IReadOnlyList<DxbcSignatureElement> signature,
        IReadOnlyList<Variable> variables, HashSet<uint> used, bool isHull, int locationOffset)
    {
        (int outer, int inner, uint outerSystem, uint innerSystem) = domain switch
        {
            DxbcTessellationDomain.Isoline => (2, 0, LineDensity, 0u),
            DxbcTessellationDomain.Triangle => (3, 1, TriangleEdge, TriangleInside),
            DxbcTessellationDomain.Quad => (4, 2, QuadEdge, QuadInside),
            _ => throw Unsupported("unknown tessellation domain")
        };
        var userFields = ReadFields(signature.Where(e => e.SystemValue == 0), variables, used, input: !isHull, isPatch: true, locationOffset);
        var factors = new List<Field>(userFields.Length + 2);
        var seen = new HashSet<BuiltIn>();
        for (int index = 0, user = 0; index < signature.Count; index++)
        {
            var field = signature[index];
            if (field.SystemValue == 0)
            {
                factors.Add(userFields[user++]);
                continue;
            }
            if (field.SystemValue == outerSystem || (domain == DxbcTessellationDomain.Isoline && field.SystemValue == LineDetail))
                AddFactors(BuiltIn.TessLevelOuter, outer, "Edge", "SV_TessFactor");
            else if (field.SystemValue == innerSystem && inner > 0)
                AddFactors(BuiltIn.TessLevelInner, inner, "Inside", "SV_InsideTessFactor");
            else throw Unsupported("unknown patch-constant system value");
        }
        if (seen.Count != (inner == 0 ? 1 : 2)) throw Unsupported("incomplete tessellation factor signature");
        return factors.ToArray();

        void AddFactors(BuiltIn builtIn, int count, string name, string semantic)
        {
            if (!seen.Add(builtIn)) return;
            var elements = signature.Where(e => string.Equals(e.SemanticName, semantic, StringComparison.OrdinalIgnoreCase)).ToArray();
            if (elements.Length != count || !elements.Select(e => e.SemanticIndex).Order().SequenceEqual(Enumerable.Range(0, count).Select(i => (uint)i)) ||
                elements.Any(e => e.ComponentType != 3 || e.Mask != 1))
                throw Unsupported($"unexpected {semantic} shape");
            Variable? variable = variables.SingleOrDefault(v => v.Patch && v.Input != isHull && v.BuiltIn == builtIn);
            if (variable is not null && (variable.Type != "float" || variable.Width != 1 || variable.Count < count || !used.Add(variable.Id)))
                throw Unsupported($"unexpected SPIR-V {semantic} shape");
            if (variable is null && (isHull || elements.Any(e => (e.Mask & e.UsedMask) != 0)))
                throw Unsupported($"missing SPIR-V {semantic}");
            factors.Add(new(name, semantic, "float", count, variable));
        }
    }

    private static void WriteStruct(StringBuilder text, string name, Field[] fields)
    {
        text.AppendLine($"struct {name}").AppendLine("{");
        foreach (Field field in fields)
        {
            string dimension = field.Count != 0 ? $"[{field.Count}]" : "";
            text.AppendLine($"    {field.Type} {field.Name}{dimension} : {field.Semantic};");
        }
        text.AppendLine("};").AppendLine();
    }

    private static string ReadSwizzle(Field field) => !char.IsAsciiDigit(field.Type[^1]) || field.Type[^1] - '0' == field.Variable!.Width
        ? "" : "." + "xyzw"[..(int)field.Variable.Width];

    private static string Type(string scalar, int width) => width == 1 ? scalar : scalar + width;
    private static string Parameters(params string[] parts) => string.Join(", ", parts.Where(p => p.Length != 0));
}
