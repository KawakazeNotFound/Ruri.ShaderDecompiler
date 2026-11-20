using System.Text;
using static Ruri.ShaderTools.Pipeline.Frontend.DxbcContainer;

namespace Ruri.ShaderTools.Pipeline.Frontend;

internal enum DxbcSignatureKind { Input, Output, PatchConstant }

internal readonly record struct DxbcSignatureElement(
    string SemanticName, uint SemanticIndex, uint SystemValue, uint ComponentType,
    uint Register, byte Mask, byte UsedMask, uint Stream = 0, uint MinPrecision = 0);

internal static class DxbcSignature
{
    public static IReadOnlyList<DxbcSignatureElement> Read(ReadOnlySpan<byte> binary, DxbcSignatureKind kind)
    {
        const uint input = 0x4e475349; // ISGN
        const uint inputExtended = 0x31475349; // ISG1
        const uint output = 0x4e47534f; // OSGN
        const uint outputExtended = 0x3147534f; // OSG1
        const uint outputWithStream = 0x3547534f; // OSG5
        const uint patch = 0x47534350; // PCSG
        const uint patchExtended = 0x31475350; // PSG1
        // All three kinds use the same record, with optional stream and precision words.
        ReadOnlySpan<uint> tags = kind switch
        {
            DxbcSignatureKind.Input => [inputExtended, input],
            DxbcSignatureKind.Output => [outputExtended, outputWithStream, output],
            DxbcSignatureKind.PatchConstant => [patchExtended, patch],
            _ => throw new ArgumentOutOfRangeException(nameof(kind))
        };
        ReadOnlySpan<byte> data = FindChunk(binary, tags, out uint tag);
        if (tag == 0) return [];

        const int headerLength = 2 * sizeof(uint);
        if (data.Length < headerLength) throw new InvalidDataException("Missing DXBC signature header.");

        bool extended = tag is inputExtended or outputExtended or patchExtended;
        bool hasStream = extended || tag == outputWithStream;
        int streamLength = hasStream ? sizeof(uint) : 0;
        int recordLength = 6 * sizeof(uint) + streamLength + (extended ? sizeof(uint) : 0);
        uint count = ReadWord(data);
        if (count > (data.Length - headerLength) / recordLength)
            throw new InvalidDataException("Invalid DXBC signature element count.");

        var elements = new DxbcSignatureElement[(int)count];
        int recordsEnd = headerLength + elements.Length * recordLength;
        for (int index = 0; index < elements.Length; index++)
        {
            var record = data.Slice(headerLength + index * recordLength, recordLength);
            var fields = record[streamLength..];
            uint nameOffset = ReadWord(fields);
            if (nameOffset < recordsEnd || nameOffset >= data.Length)
                throw new InvalidDataException("Invalid DXBC signature semantic name offset.");
            ReadOnlySpan<byte> name = data[(int)nameOffset..];
            int end = name.IndexOf((byte)0);
            if (end <= 0) throw new InvalidDataException("Invalid DXBC signature semantic name.");
            elements[index] = new DxbcSignatureElement(
                Encoding.ASCII.GetString(name[..end]), ReadWord(fields[4..]), ReadWord(fields[8..]),
                ReadWord(fields[12..]), ReadWord(fields[16..]), fields[20], fields[21],
                hasStream ? ReadWord(record) : 0, extended ? ReadWord(fields[24..]) : 0);
        }
        return elements;
    }
}
