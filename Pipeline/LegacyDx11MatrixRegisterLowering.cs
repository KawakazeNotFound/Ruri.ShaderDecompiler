using Ruri.ShaderTools.Spirv;
using Ruri.ShaderTools.Spirv.Analysis;

namespace Ruri.ShaderTools.Pipeline;

// Reconstruct original flat register ROW reads after the structurer introduced
// RowMajor matrix storage but retargeted the register number as a logical COLUMN.
internal static class LegacyDx11MatrixRegisterLowering
{
    private sealed record Path(uint Variable, uint Storage, uint RootType, List<uint> Indices);
    private sealed record MatrixRead(Path Path, int IndexAt, uint ScalarType, uint VectorType);

    public static byte[] Apply(byte[] before, byte[] structured)
    {
        SpirvModule previous = SpirvModule.Parse(before), module = SpirvModule.Parse(structured);
        var oldStructs = previous.Instructions.Where(i => i.OpCode == 30).Select(i => i[1]).ToHashSet();
        var generated = module.Instructions.Where(i => i.OpCode == 72 && i.WordCount == 4
            && i[3] == 4 && !oldStructs.Contains(i[1])).Select(i => (i[1], i[2])).ToHashSet();
        if (generated.Count == 0) return structured;
        var definitions = new Dictionary<uint, SpirvInstruction>();
        foreach (SpirvInstruction i in module.Instructions)
        {
            int? slot = SpvInstructionTraits.GetResultIdIndex(i);
            if (slot.HasValue) definitions.Add(i[slot.Value], i);
        }
        uint Literal(uint id)
        {
            if (!definitions.TryGetValue(id, out var c) || c.OpCode != 43 || c.WordCount != 4)
                throw new InvalidOperationException("Legacy matrix register lowering: UNKNOWN nonconstant struct member.");
            return c[3];
        }
        Path? Origin(uint id, HashSet<uint> seen)
        {
            if (!seen.Add(id) || !definitions.TryGetValue(id, out var instruction)) return null;
            if (instruction.OpCode == 59 && instruction.WordCount >= 4)
            {
                var pointer = definitions[instruction[1]];
                return pointer.OpCode == 32 && pointer.WordCount == 4
                    ? new Path(id, pointer[2], pointer[3], new List<uint>()) : null;
            }
            if (instruction.OpCode is not (65 or 66) || instruction.WordCount < 5) return null;
            Path? parent = Origin(instruction[3], seen);
            if (parent is null) return null;
            return parent with { Indices = parent.Indices.Concat(instruction.Words[4..].ToArray()).ToList() };
        }
        MatrixRead? Analyze(Path path)
        {
            uint typeId = path.RootType;
            bool generatedMember = false;
            for (int at = 0; at <= path.Indices.Count; at++)
            {
                var type = definitions[typeId];
                if (type.OpCode == 24 && generatedMember)
                {
                    var vector = definitions[type[2]];
                    var scalar = definitions[vector[2]];
                    if (path.Storage != 2 || type.WordCount != 4 || type[3] != 4
                        || vector.OpCode != 23 || vector.WordCount != 4 || vector[3] != 4
                        || scalar.OpCode != 22 || scalar.WordCount != 3 || scalar[2] != 32
                        || path.Indices.Count - at is not (0 or 1 or 2))
                        throw new InvalidOperationException("Legacy matrix register lowering: UNKNOWN matrix shape/storage/access; only uniform float4x4 row reads are supported.");
                    return new MatrixRead(path, at, scalar[1], vector[1]);
                }
                if (at == path.Indices.Count) return null;
                if (type.OpCode == 30)
                {
                    uint member = Literal(path.Indices[at]);
                    if (member >= type.WordCount - 2) throw new InvalidOperationException("Invalid matrix struct member.");
                    generatedMember = generated.Contains((typeId, member));
                    typeId = type[(int)member + 2];
                }
                else if (type.OpCode is 28 or 29) typeId = type[2];
                else return null;
            }
            return null;
        }
        var pointers = new Dictionary<uint, MatrixRead>();
        foreach (SpirvInstruction i in module.Instructions)
        {
            if (i.OpCode is not (65 or 66)) continue;
            Path? origin = Origin(i[2], new HashSet<uint>());
            MatrixRead? read = origin is null ? null : Analyze(origin);
            if (read is not null) pointers.Add(i[2], read);
        }
        if (pointers.Count == 0) return structured;
        foreach (SpirvInstruction i in module.Instructions)
        {
            // Annotation arguments include literal offsets, bindings and member numbers.
            // Only target-id slots count; a literal matching an SSA id is not decoration.
            if (i.OpCode is 71 or 72 && i.WordCount >= 2 && pointers.ContainsKey(i[1]))
                throw new InvalidOperationException("Legacy matrix register lowering: UNKNOWN decorated matrix pointer.");
            if (i.OpCode is 74 or 75)
                for (int at = 2; at < i.WordCount; at += i.OpCode == 75 ? 2 : 1)
                    if (pointers.ContainsKey(i[at]))
                        throw new InvalidOperationException("Legacy matrix register lowering: UNKNOWN decorated matrix pointer.");
        }
        SpirvInstruction Emit(ushort op, params uint[] args)
        {
            uint[] words = new uint[args.Length + 1];
            words[0] = SpvOpCode.MakeInstructionWord(op, (ushort)words.Length);
            args.CopyTo(words, 1);
            return module.CreateInstruction(op, words);
        }
        uint Pointer(uint type)
        {
            var existing = module.Instructions.FirstOrDefault(i => i.OpCode == 32 && i.WordCount == 4 && i[2] == 2 && i[3] == type);
            if (existing is not null) return existing[1];
            uint id = module.AllocateId(); module.AppendType(Emit(32, id, 2, type)); return id;
        }
        var uintType = module.Instructions.FirstOrDefault(i => i.OpCode == 21 && i.WordCount == 4 && i[2] == 32 && i[3] == 0);
        uint uintId = uintType is null ? module.AllocateId() : uintType[1];
        if (uintType is null) module.AppendType(Emit(21, uintId, 32, 0));
        uint Constant(uint value)
        {
            var old = module.Instructions.FirstOrDefault(i => i.OpCode == 43 && i.WordCount == 4 && i[1] == uintId && i[3] == value);
            if (old is not null) return old[2];
            uint id = module.AllocateId(); module.AppendType(Emit(43, uintId, id, value)); return id;
        }
        uint[] columns = Enumerable.Range(0, 4).Select(c => Constant((uint)c)).ToArray();
        var loads = module.Instructions.Where(i => i.OpCode == 61 && i.WordCount >= 4 && pointers.ContainsKey(i[3])).ToArray();
        foreach (SpirvInstruction load in loads)
        {
            MatrixRead read = pointers[load[3]];
            bool scalarRead = read.Path.Indices.Count - read.IndexAt == 2;
            if (load.WordCount != 4 || load[1] != (scalarRead ? read.ScalarType : read.VectorType))
                throw new InvalidOperationException("Legacy matrix register lowering: UNKNOWN matrix load shape/memory operands.");
            uint pointerType = Pointer(read.ScalarType);
            uint[] prefix = read.Path.Indices.Take(read.IndexAt).ToArray();
            uint row = read.Path.Indices[read.IndexAt];
            var additions = new List<SpirvInstruction>();
            uint ScalarPointer(uint column)
            {
                uint pointer = module.AllocateId();
                additions.Add(Emit(65, new uint[] { pointerType, pointer, read.Path.Variable }
                    .Concat(prefix).Concat(new uint[] { column, row }).ToArray()));
                return pointer;
            }
            if (scalarRead)
                load[3] = ScalarPointer(read.Path.Indices[read.IndexAt + 1]);
            else
            {
                var values = new List<uint>();
                foreach (uint column in columns)
                {
                    uint pointer = ScalarPointer(column), value = module.AllocateId();
                    additions.Add(Emit(61, read.ScalarType, value, pointer)); values.Add(value);
                }
                load.OpCode = 80;
                load.SetWords(new uint[] { SpvOpCode.MakeInstructionWord(80, 7), read.VectorType, load[2] }.Concat(values).ToArray());
            }
            module.Instructions.InsertRange(module.Instructions.IndexOf(load), additions);
        }
        // Retire old logical-column chains only if every data consumer was lowered.
        LiveUseCounter live = LiveUseCounter.Build(module);
        foreach (uint id in pointers.Keys.Reverse())
        {
            if (live.HasLiveConsumer(id))
                throw new InvalidOperationException("Legacy matrix register lowering: UNKNOWN matrix pointer escape/unlowered use.");
            live.Retire(definitions[id]);
            foreach (SpirvInstruction i in module.Instructions)
                if (i.OpCode == 5 && i.WordCount >= 2 && i[1] == id) i.MakeNop();
        }
        return module.ToBytes();
    }
}
