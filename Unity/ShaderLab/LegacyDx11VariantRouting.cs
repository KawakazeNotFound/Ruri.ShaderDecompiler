using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Ruri.ShaderTools.Unity.ShaderLab;

public static class LegacyDx11VariantRouting
{
    internal sealed record StageRoute(List<ushort> Keywords, List<IGrouping<string, UnitySerializedSubProgram>> Groups, int DefaultIndex);

    internal static List<string> EffectiveKeywordNames(List<string> shaderNames, UnitySerializedPass pass)
    {
        if (shaderNames.Count > 0)
            return shaderNames;

        List<string> names = [];
        foreach (UnitySerializedNameIndex entry in pass.NameIndices)
        {
            while (names.Count <= entry.Second)
                names.Add(string.Empty);
            names[entry.Second] = entry.First;
        }
        return names;
    }

    internal static StageRoute BuildStage(List<UnitySerializedSubProgram> subPrograms)
    {
        var groups = subPrograms
            .GroupBy(program => string.Join(",", program.KeywordIndices.OrderBy(index => index)))
            .ToList();
        int defaultIndex = groups.FindIndex(group => group.First().KeywordIndices.Count == 0);
        return new(KeywordSymbols.DistinctIndices(subPrograms), groups, Math.Max(defaultIndex, 0));
    }

    public static bool CanCatalogStage(UnitySerializedProgram program, IReadOnlyList<string> keywordNames)
    {
        var indices = KeywordSymbols.DistinctIndices(program.SubPrograms);
        return indices.Count > 0 && indices.All(index => index < keywordNames.Count && IsVariantMacro(keywordNames[index]))
            && indices.Select(index => keywordNames[index]).Distinct(StringComparer.Ordinal).Count() == indices.Count;
    }

    internal static Dictionary<UnitySerializedSubProgram, string>? ProveVariantFamilies(List<string> shaderNames, UnitySerializedPass pass)
    {
        var vertex = BuildStage(pass.ProgVertex!.SubPrograms);
        var fragment = BuildStage(pass.ProgFragment!.SubPrograms);
        if (vertex.Groups.Any(group => group.Count() != 1) || fragment.Groups.Any(group => group.Count() != 1))
            return null;

        var names = EffectiveKeywordNames(shaderNames, pass);
        var macros = new HashSet<string>(StringComparer.Ordinal);
        foreach (ushort index in vertex.Keywords.Union(fragment.Keywords))
        {
            if (index >= names.Count || String.IsNullOrWhiteSpace(names[index]))
                return null;
            string symbol = KeywordSymbols.ToSymbol(names, index);
            if (!IsVariantMacro(symbol) || !macros.Add(symbol))
                return null;
        }

        var shared = vertex.Keywords.Intersect(fragment.Keywords).ToHashSet();
        string Projection(UnitySerializedSubProgram program)
            => string.Join(",", program.KeywordIndices.Where(shared.Contains).OrderBy(index => index));
        string defaultProjection = Projection(vertex.Groups[vertex.DefaultIndex].First());
        if (Projection(fragment.Groups[fragment.DefaultIndex].First()) != defaultProjection)
            return null;

        var vertexFamilies = vertex.Groups.Select(group => group.First()).ToLookup(Projection, StringComparer.Ordinal);
        var fragmentFamilies = fragment.Groups.Select(group => group.First()).ToLookup(Projection, StringComparer.Ordinal);
        if (vertexFamilies.Count != fragmentFamilies.Count || vertexFamilies.Any(family => !fragmentFamilies.Contains(family.Key)))
            return null;

        // A non-default projection must select an explicit branch for every local assignment.
        // Missing shared projections and missing local assignments of the default projection fall back together.
        if (!HasCompleteNonDefaultFamilies(vertexFamilies, defaultProjection, vertex.Keywords.Count - shared.Count)
            || !HasCompleteNonDefaultFamilies(fragmentFamilies, defaultProjection, fragment.Keywords.Count - shared.Count))
            return null;
        return vertexFamilies.Concat(fragmentFamilies)
            .SelectMany(family => family.Select(program => (Program: program, Family: family.Key)))
            .ToDictionary(pair => pair.Program, pair => pair.Family);
    }

    private static bool HasCompleteNonDefaultFamilies(ILookup<string, UnitySerializedSubProgram> families, string fallback, int localCount)
        => families.All(family => family.Key == fallback || (localCount < 31 && family.Count() == (1 << localCount)));

    private static bool IsVariantMacro(string symbol)
        => Regex.IsMatch(symbol, @"\A[A-Za-z_][A-Za-z_0-9]*\z", RegexOptions.CultureInvariant)
            && symbol is not ("_" or "defined" or "true" or "false" or "SHADER_TARGET" or "UNITY_VERSION")
            && !symbol.StartsWith("__", StringComparison.Ordinal)
            && !symbol.StartsWith("SHADER_STAGE_", StringComparison.Ordinal)
            && !symbol.StartsWith("SHADER_API_", StringComparison.Ordinal)
            && !symbol.StartsWith("UNITY_COMPILER_", StringComparison.Ordinal)
            && !symbol.EndsWith("_INCLUDED", StringComparison.Ordinal);
}
