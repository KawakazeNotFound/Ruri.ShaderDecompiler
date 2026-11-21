namespace Ruri.ShaderTools.Unity.ShaderLab;

internal static class VariantKeywordPragmas
{
    private sealed record KeywordGroup(int StageMask, List<string> Keywords);

    public static void Write(IndentedWriter writer, UnitySerializedPass pass, List<string> names,
        List<string> passKeywords, string? engineKeyword, ShaderLabKeywordLayout layout)
    {
        var keywords = passKeywords.Where(keyword => keyword != engineKeyword).ToArray();
        var independent = keywords.Select(keyword => $"#pragma multi_compile _ {keyword}").ToArray();
        if (layout == ShaderLabKeywordLayout.Independent || keywords.Length == 0)
        {
            WriteLines(independent);
            return;
        }

        var stages = pass.EnumerateProgramSlots().Where(slot => slot.Program.SubPrograms.Count > 0).ToArray();
        var indices = stages.SelectMany(slot => KeywordSymbols.DistinctIndices(slot.Program.SubPrograms)).Distinct().ToArray();
        if (stages.Any(slot => slot.Program.SubPrograms.Any(program => program.KeywordIndices.Count > 0)
                && !LegacyDx11VariantRouting.CanCatalogStage(slot.Program, names))
            || indices.Length != passKeywords.Count)
        {
            WriteLines(independent);
            return;
        }

        var stageMasks = keywords.ToDictionary(keyword => keyword, _ => 0, StringComparer.Ordinal);
        var cooccurrence = keywords.ToDictionary(keyword => keyword, _ => new HashSet<string>(StringComparer.Ordinal),
            StringComparer.Ordinal);
        for (int stageIndex = 0; stageIndex < stages.Length; stageIndex++)
        {
            foreach (var program in stages[stageIndex].Program.SubPrograms)
            {
                var active = program.KeywordIndices.Select(index => names[index]).Where(keyword => keyword != engineKeyword).ToArray();
                foreach (string keyword in active)
                {
                    stageMasks[keyword] |= 1 << stageIndex;
                    cooccurrence[keyword].UnionWith(active);
                }
            }
        }

        var groups = new List<KeywordGroup>();
        foreach (string keyword in keywords)
        {
            int mask = stageMasks[keyword];
            var group = groups.FirstOrDefault(group => group.StageMask == mask
                && !group.Keywords.Any(cooccurrence[keyword].Contains));
            if (group is null)
                groups.Add(new(mask, [keyword]));
            else
                group.Keywords.Add(keyword);
        }

        bool scopeStages = layout == ShaderLabKeywordLayout.CacheStageGroups && stages.Length == 2
            && stages.All(slot => slot.Stage is "Vertex" or "Fragment");
        var restored = groups.Select(group =>
        {
            string suffix = string.Empty;
            if (scopeStages && group.StageMask is 1 or 2)
                suffix = "_" + stages[group.StageMask == 1 ? 0 : 1].Stage.ToLowerInvariant();
            return $"#pragma multi_compile{suffix} _ {string.Join(' ', group.Keywords)}";
        }).ToArray();
        if (restored.SequenceEqual(independent))
        {
            WriteLines(independent);
            return;
        }

        writer.Line("#if defined(SHADER_API_D3D11)");
        WriteLines(restored);
        writer.Line("#else");
        WriteLines(independent);
        writer.Line("#endif");

        void WriteLines(IEnumerable<string> lines)
        {
            foreach (string line in lines)
                writer.Line(line);
        }
    }
}
