using Ruri.ShaderTools.Spirv;

namespace Ruri.ShaderTools.Pipeline.Frontend;

internal static class DxbcComputeRegisterSpace
{
    public const int UnorderedAccessSet = 1;

    public static byte[] PreserveUavDeclaration(byte[] spirv)
    {
        var module = SpirvModule.Parse(spirv);
        var uavs = new HashSet<uint>();
        foreach (var instruction in module.Instructions)
        {
            var words = instruction.Words;
            if (instruction.OpCode == SpvOpCode.OpDecorate && words.Length == 4 &&
                words[2] == Decoration.DescriptorSet && words[3] == UnorderedAccessSet)
            {
                uavs.Add(words[1]);
            }
        }
        // Even a UAV this kernel only reads must be declared as uN rather than tN in the HLSL ABI.
        int removed = module.Instructions.RemoveAll(instruction => instruction.OpCode == SpvOpCode.OpDecorate &&
            instruction.Words.Length == 3 && instruction.Words[2] == Decoration.NonWritable && uavs.Contains(instruction.Words[1]));
        return removed == 0 ? spirv : module.ToBytes();
    }
}
