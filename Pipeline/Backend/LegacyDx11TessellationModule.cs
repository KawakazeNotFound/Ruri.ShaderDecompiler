using System.Runtime.InteropServices;
using Ruri.ShaderTools.Pipeline.Frontend;
using Ruri.ShaderTools.Spirv;
using Silk.NET.SPIRV;
using static Ruri.ShaderTools.Spirv.SpvOpCode;
using StorageClass = Ruri.ShaderTools.Spirv.StorageClass;

namespace Ruri.ShaderTools.Pipeline.Backend;

internal sealed class LegacyDx11TessellationModule
{
    internal record Variable(uint Id, bool Input, bool Patch, uint? Location, BuiltIn? BuiltIn, uint Count, uint Width, string Type, string? OriginalName)
    {
        public string Name => "RuriTessellationIo" + Id;
    }

    public const string Control = "RuriTessellationControl";
    public const string Patch = "RuriTessellationPatch";
    public const string Evaluate = "RuriTessellationEvaluate";
    private const uint PrivateStorage = (uint)Silk.NET.SPIRV.StorageClass.Private;

    public required byte[] Bytes { get; init; }
    public required IReadOnlyList<Variable> Variables { get; init; }
    public required EntryPointSelection Entry { get; init; }

    public static LegacyDx11TessellationModule Lower(byte[] spirv, DxbcTessellation declarations)
    {
        bool isHull = declarations.Stage == PipelineStage.TessControl;
        var module = SpirvModule.Parse(spirv);
        var instructions = module.Instructions;
        var entries = instructions.Where(i => i.OpCode == OpEntryPoint).ToArray();
        uint expected = isHull ? (uint)ExecutionModel.TessellationControl : (uint)ExecutionModel.TessellationEvaluation;
        if (entries.Length != 1 || entries[0][1] != expected)
            throw Unsupported("expected a single entry of the DXBC stage");
        SpirvInstruction entry = entries[0];
        uint originalEntryId = entry[2];

        var definitions = instructions.Where(i => SpvInstructionTraits.GetResultIdIndex(i).HasValue)
            .ToDictionary(i => i[SpvInstructionTraits.GetResultIdIndex(i)!.Value]);
        var decorations = instructions.Where(i => i.OpCode == OpDecorate).ToLookup(i => i[1]);
        var originalNames = instructions.Where(i => i.OpCode == OpName).GroupBy(i => i[1])
            .ToDictionary(group => group.Key, group => SpirvLiteral.ReadString(group.Last().Words, 2));
        var globalVariables = instructions.Where(i => i.OpCode == OpVariable && i[3] is StorageClass.Input or StorageClass.Output).ToArray();
        var variables = globalVariables.Select(i => ReadVariable(i, definitions, decorations, originalNames.GetValueOrDefault(i[2]))).ToArray();
        var byId = variables.ToDictionary(v => v.Id);
        var names = new Dictionary<uint, string>();
        foreach (Variable variable in variables) names.Add(variable.Id, variable.Name);

        int start = instructions.FindIndex(i => i.OpCode == OpFunction && i[2] == originalEntryId);
        if (start < 0) throw Unsupported("missing SPIR-V entry definition");
        int end = instructions.FindIndex(start + 1, i => i.OpCode == OpFunctionEnd);
        if (end < 0) throw Unsupported("unterminated SPIR-V entry definition");
        SpirvInstruction function = instructions[start];
        uint carrierId;
        uint[] bodies;
        if (isHull)
        {
            bodies = ReadHullWrapper(instructions.GetRange(start + 1, end - start - 1), definitions, byId, declarations.OutputControlPoints);
            RemoveFrontendFactorClamp(instructions, definitions, variables, bodies[1], declarations);
            names.Add(bodies[0], Control);
            names.Add(bodies[1], Patch);
            carrierId = originalEntryId;
            instructions.RemoveRange(start, end - start + 1);
        }
        else
        {
            bodies = [originalEntryId];
            names.Add(originalEntryId, Evaluate);
            carrierId = module.AllocateId();
        }

        var patchedNames = new HashSet<string>(names.Values, StringComparer.Ordinal);
        foreach (SpirvInstruction instruction in instructions)
        {
            switch (instruction.OpCode)
            {
                case OpName when names.ContainsKey(instruction[1]):
                    instruction.MakeNop();
                    break;
                case OpName when patchedNames.Contains(SpirvLiteral.ReadString(instruction.Words, 2)):
                    throw Unsupported("generated SPIR-V name collision");
                case OpTypePointer when instruction[2] is StorageClass.Input or StorageClass.Output:
                    instruction[2] = PrivateStorage;
                    break;
                case OpVariable when byId.ContainsKey(instruction[2]):
                    instruction[3] = PrivateStorage;
                    break;
                case OpExecutionMode when instruction[1] == originalEntryId:
                    if ((ExecutionMode)instruction[2] is ExecutionMode.DenormPreserve or ExecutionMode.DenormFlushToZero
                        or ExecutionMode.SignedZeroInfNanPreserve or ExecutionMode.RoundingModeRte or ExecutionMode.RoundingModeRtz)
                        instruction[1] = carrierId;
                    else instruction.MakeNop();
                    break;
                case OpCapability when instruction[1] == (uint)Capability.Tessellation:
                    instruction.MakeNop();
                    break;
                case OpDecorate when byId.ContainsKey(instruction[1]):
                    if (instruction[2] != (uint)Silk.NET.SPIRV.Decoration.Invariant) instruction.MakeNop();
                    break;
            }
        }

        string carrierName = SpirvLiteral.ReadString(entry.Words, 3);
        int nameBytes = MemoryMarshal.AsBytes((ReadOnlySpan<uint>)entry.Words[3..]).IndexOf((byte)0);
        if (nameBytes < 0) throw Unsupported("malformed SPIR-V entry name");
        int interfaceStart = 3 + (nameBytes / sizeof(uint)) + 1;
        uint[] interfaceIds = entry.Words[interfaceStart..].ToArray();
        if (module.Header[1] < 0x10400)
            interfaceIds = interfaceIds.Where(id => !byId.ContainsKey(id)).ToArray();
        var carrierEntry = new uint[interfaceStart + interfaceIds.Length];
        entry.Words[..interfaceStart].CopyTo(carrierEntry);
        carrierEntry[0] = MakeInstructionWord(OpEntryPoint, (ushort)carrierEntry.Length);
        carrierEntry[1] = (uint)ExecutionModel.Vertex;
        carrierEntry[2] = carrierId;
        interfaceIds.CopyTo(carrierEntry, interfaceStart);
        entry.SetWords(carrierEntry);

        var bodyFunctions = bodies.Select(id => definitions[id]).ToArray();
        if (bodyFunctions.Any(f => f.OpCode != OpFunction || f[1] != function[1] || f[4] != function[4]))
            throw Unsupported("expected void/no-argument tessellation bodies");
        AddInstruction(OpFunction, function[1], carrierId, 0, function[4]);
        AddInstruction(OpLabel, module.AllocateId());
        foreach (uint body in bodies) AddInstruction(OpFunctionCall, function[1], module.AllocateId(), body);
        AddInstruction((ushort)Op.Return);
        AddInstruction(OpFunctionEnd);
        foreach (var pair in names) module.InsertDebugName(pair.Key, pair.Value);
        return new() { Bytes = module.ToBytes(), Variables = variables, Entry = new(PipelineStage.Vertex, carrierName, (uint)ExecutionModel.Vertex) };

        void AddInstruction(ushort opcode, params uint[] operands)
        {
            var words = new uint[1 + operands.Length];
            words[0] = MakeInstructionWord(opcode, (ushort)words.Length);
            operands.CopyTo(words, 1);
            instructions.Add(module.CreateInstruction(opcode, words));
        }
    }

    private static uint[] ReadHullWrapper(List<SpirvInstruction> instructions,
        Dictionary<uint, SpirvInstruction> definitions, Dictionary<uint, Variable> variables, int pointCount)
    {
        // dxil-spirv's build_hull_main calls two leaves, with the patch call guarded by invocation zero when N > 1.
        var calls = instructions.Where(i => i.OpCode == OpFunctionCall).ToArray();
        var represented = instructions.Where(i => i.OpCode is not (OpNop or OpLabel or OpBranch or OpLine)).ToArray();
        bool HasOpcodes(params Op[] opcodes) => represented.Select(i => i.OpCode).SequenceEqual(opcodes.Select(op => (ushort)op));
        if (calls.Length != 2 || calls.Any(i => i.WordCount != 4) || calls[0][3] == calls[1][3])
            throw Unsupported("unproven hull control and patch bodies");
        if (pointCount == 1)
        {
            if (!HasOpcodes(Op.FunctionCall, Op.FunctionCall, Op.Return))
                throw Unsupported("unproven single-control-point hull wrapper");
        }
        else
        {
            if (!HasOpcodes(Op.FunctionCall, Op.Load, Op.IEqual, Op.ControlBarrier, Op.SelectionMerge, Op.BranchConditional, Op.FunctionCall, Op.Return))
                throw Unsupported("unproven multi-control-point hull wrapper");
            SpirvInstruction load = represented[1];
            SpirvInstruction compare = represented[2];
            SpirvInstruction branch = represented[5];
            int patchCallIndex = instructions.IndexOf(calls[1]);
            SpirvInstruction merge = represented[4];
            if (!variables.TryGetValue(load[3], out var variable) || variable.BuiltIn != BuiltIn.InvocationId ||
                compare[3] != load[2] || !definitions.TryGetValue(compare[4], out SpirvInstruction? zero) ||
                zero.OpCode != OpConstant || zero[3] != 0 || branch[1] != compare[2] || branch[3] != merge[1] ||
                patchCallIndex < 1 || instructions[patchCallIndex - 1].OpCode != OpLabel ||
                instructions[patchCallIndex - 1][1] != branch[2] || instructions[patchCallIndex + 1].OpCode != OpBranch ||
                instructions[patchCallIndex + 1][1] != merge[1])
                throw Unsupported("hull patch call is not proven uniform");
        }
        return [calls[0][3], calls[1][3]];
    }

    private static void RemoveFrontendFactorClamp(List<SpirvInstruction> instructions, Dictionary<uint, SpirvInstruction> definitions,
        IReadOnlyList<Variable> variables, uint patchBody, DxbcTessellation declarations)
    {
        // dxbc-spirv's IoMap::applyMaxTessFactor appends an output load/clamp/store for Vulkan.
        // D3D11's domain stage can read raw factors, and its tessellator already enforces the declared limit.
        var (outer, inner) = declarations.Domain switch
        {
            DxbcTessellationDomain.Isoline => (2, 0),
            DxbcTessellationDomain.Triangle => (3, 1),
            DxbcTessellationDomain.Quad => (4, 2),
            _ => throw Unsupported("unknown domain for the tessellation factor clamp")
        };
        var expected = new HashSet<(uint Id, uint Index)>();
        Add(BuiltIn.TessLevelOuter, outer);
        if (inner != 0) Add(BuiltIn.TessLevelInner, inner);
        uint maximum = BitConverter.SingleToUInt32Bits(declarations.MaximumFactor ?? 64.0f);
        int start = instructions.IndexOf(definitions[patchBody]);
        int end = instructions.FindIndex(start + 1, i => i.OpCode == OpFunctionEnd);
        var patch = instructions.GetRange(start + 1, end - start - 1)
            .Where(i => i.OpCode is not (OpNop or OpLine)).ToArray();
        int cursor = patch.Length - 1;
        if (patch[cursor--].OpCode != (ushort)Op.Return)
            throw Unsupported("unproven tail of the hull patch-constant function");
        var removals = new List<SpirvInstruction>();
        while (expected.Count > 0)
        {
            if (cursor < 3) throw Unsupported("missing frontend tessellation factor clamp");
            SpirvInstruction store = patch[cursor--];
            if (store.OpCode != OpStore || store.WordCount != 3 || !ReadFactorAccess(store[1], out var component))
                throw Unsupported("unproven write of a frontend tessellation factor clamp");
            SpirvInstruction last = patch[cursor];
            SpirvInstruction? storeAccess = null;
            if (last.OpCode == OpAccessChain && last[2] == store[1]) storeAccess = patch[cursor--];
            SpirvInstruction clamp = patch[cursor--];
            SpirvInstruction load = patch[cursor--];
            SpirvInstruction loadAccess = patch[cursor--];
            if (!expected.Remove(component) || clamp.OpCode != OpExtInst || clamp.WordCount != 8 ||
                definitions[clamp[3]].OpCode != OpExtInstImport || SpirvLiteral.ReadString(definitions[clamp[3]].Words, 2) != "GLSL.std.450" ||
                clamp[4] != 81 || clamp[2] != store[2] || !ConstantIs(clamp[6], 0, OpTypeFloat) || !ConstantIs(clamp[7], maximum, OpTypeFloat) ||
                load.OpCode != OpLoad || load.WordCount != 4 || load[2] != clamp[5] ||
                loadAccess.OpCode != OpAccessChain || loadAccess[2] != load[3] || !ReadFactorAccess(load[3], out var read) || read != component)
                throw Unsupported("unproven data flow of a frontend tessellation factor clamp");
            removals.AddRange([store, clamp, load, loadAccess]);
            if (storeAccess is not null) removals.Add(storeAccess);
        }
        foreach (SpirvInstruction instruction in removals) instruction.MakeNop();

        void Add(BuiltIn builtIn, int count)
        {
            Variable? variable = variables.SingleOrDefault(v => !v.Input && v.Patch && v.BuiltIn == builtIn);
            if (variable is null) throw Unsupported($"missing {builtIn} output");
            for (uint index = 0; index < count; index++) expected.Add((variable.Id, index));
        }

        bool ConstantIs(uint id, uint value, ushort typeOpcode) => definitions.TryGetValue(id, out var constant) &&
            constant.OpCode == OpConstant && constant.WordCount == 4 && constant[3] == value &&
            definitions[constant[1]].OpCode == typeOpcode && definitions[constant[1]][2] == 32;

        bool ReadFactorAccess(uint id, out (uint, uint) value)
        {
            value = default;
            if (!definitions.TryGetValue(id, out var access) || access.OpCode != OpAccessChain || access.WordCount != 5 ||
                !definitions.TryGetValue(access[4], out var index) || index.OpCode != OpConstant || index.WordCount != 4)
                return false;
            value = (access[3], index[3]);
            return true;
        }
    }

    private static Variable ReadVariable(SpirvInstruction variable, Dictionary<uint, SpirvInstruction> definitions,
        ILookup<uint, SpirvInstruction> decorations, string? originalName)
    {
        uint id = variable[2];
        uint? location = null;
        BuiltIn? builtIn = null;
        bool patch = false;
        foreach (SpirvInstruction decoration in decorations[id])
        {
            switch ((Silk.NET.SPIRV.Decoration)decoration[2])
            {
                case Silk.NET.SPIRV.Decoration.Location: location = decoration[3]; break;
                case Silk.NET.SPIRV.Decoration.BuiltIn: builtIn = (BuiltIn)decoration[3]; break;
                case Silk.NET.SPIRV.Decoration.Patch: patch = true; break;
                case Silk.NET.SPIRV.Decoration.Invariant: break;
                default: throw Unsupported($"unknown tessellation interface decoration {decoration[2]}");
            }
        }
        if (location.HasValue == builtIn.HasValue)
            throw Unsupported($"interface variable {id} has no unique location or built-in");
        SpirvInstruction type = definitions[definitions[variable[1]][3]];
        uint count = 0;
        uint width = 1;
        if (type.OpCode == OpTypeArray)
        {
            SpirvInstruction length = definitions[type[3]];
            if (length.OpCode != OpConstant || length.WordCount != 4 || length[3] is < 1 or > 32)
                throw Unsupported("interface array does not have a literal control-point/factor dimension");
            count = length[3];
            type = definitions[type[2]];
        }
        if (type.OpCode == OpTypeVector)
        {
            width = type[3];
            type = definitions[type[2]];
        }
        string scalar = type.OpCode switch
        {
            OpTypeFloat when type[2] == 32 => "float",
            OpTypeInt when type[2] == 32 => type[3] == 0 ? "uint" : "int",
            _ => throw Unsupported($"unsupported scalar/vector type for tessellation IO {id}")
        };
        return new(id, variable[3] == StorageClass.Input, patch, location, builtIn, count, width, scalar, originalName);
    }

    internal static InvalidOperationException Unsupported(string detail) => new("DXBC tessellation: " + detail + ".");
}
