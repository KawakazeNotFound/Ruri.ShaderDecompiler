namespace Ruri.ShaderTools.Pipeline.Frontend;

/// <summary>Vertex input register and original DXBC semantic, for the existing vertex attribute remap.</summary>
internal readonly record struct InputSignatureElement(string SemanticName, uint SemanticIndex, uint Register);

internal static class DxbcInputSignature
{
    public static IReadOnlyList<InputSignatureElement> Read(ReadOnlySpan<byte> binary)
    {
        try
        {
            return DxbcSignature.Read(binary, DxbcSignatureKind.Input)
                .Select(element => new InputSignatureElement(element.SemanticName, element.SemanticIndex, element.Register)).ToArray();
        }
        catch (InvalidDataException)
        {
            return [];
        }
    }
}
