"use client";

import type { ChangeEvent, ComponentType } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { AlertCircle, ArrowRightLeft, CheckCircle2, Loader2, RefreshCw, Table2 } from "lucide-react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const getErrorMessage = (error: unknown, fallback: string) => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") {
      return detail;
    }
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return fallback;
};

export type ComparisonPaper = {
  id: string;
  title: string | null;
  year: number | null;
};

export type ComparisonEntitySummary = {
  shared: string[];
  unique: Record<string, string[]>;
};

export type ComparisonSummary = {
  methods: ComparisonEntitySummary;
  datasets: ComparisonEntitySummary;
  metrics: ComparisonEntitySummary;
  tasks: ComparisonEntitySummary;
};

export type EvidenceItem = {
  snippet?: string;
  page?: number;
  [key: string]: unknown;
};

export type ComparisonCell = {
  result_id: string;
  value_numeric: number | null;
  value_text: string | null;
  unit: string | null;
  is_sota: boolean | null;
  confidence: number | null;
  evidence: EvidenceItem[];
};

export type ComparisonRow = {
  method_id: string | null;
  method_name: string | null;
  dataset_id: string | null;
  dataset_name: string | null;
  metric_id: string | null;
  metric_name: string | null;
  metric_unit: string | null;
  task_id: string | null;
  task_name: string | null;
  split: string | null;
  papers: Record<string, ComparisonCell>;
};

export type PaperComparisonResponse = {
  papers: ComparisonPaper[];
  summary: ComparisonSummary;
  matrix: ComparisonRow[];
};

export type PaperListItem = {
  id: string;
  title: string | null;
  year: number | null;
};

const formatPaperLabel = (paper: PaperListItem) => {
  const suffix = paper.year ? ` (${paper.year})` : "";
  const title = paper.title ?? "Untitled paper";
  return `${title}${suffix}`;
};

const formatValue = (cell: ComparisonCell) => {
  if (cell.value_text) {
    return cell.value_text;
  }
  if (typeof cell.value_numeric === "number") {
    const value = cell.value_numeric.toLocaleString(undefined, {
      maximumFractionDigits: 4,
      minimumFractionDigits: Number.isInteger(cell.value_numeric) ? 0 : 2,
    });
    return cell.unit ? `${value} ${cell.unit}` : value;
  }
  return "—";
};

const EvidenceList = ({ evidence }: { evidence: EvidenceItem[] }) => {
  if (!evidence?.length) {
    return null;
  }

  return (
    <ul className="mt-2 space-y-2 rounded-md border border-dashed border-muted-foreground/20 bg-muted/30 p-3 text-xs text-muted-foreground">
      {evidence.slice(0, 3).map((item, index) => {
        const snippet = typeof item.snippet === "string" ? item.snippet : null;
        const page = typeof item.page === "number" ? item.page : undefined;
        return (
          <li key={index}>
            {snippet ? <p className="line-clamp-3">{snippet}</p> : <p>Referenced evidence snippet</p>}
            {page !== undefined ? <p className="mt-1 font-medium text-foreground">Page {page}</p> : null}
          </li>
        );
      })}
      {evidence.length > 3 ? (
        <li className="text-[11px] italic text-muted-foreground">+{evidence.length - 3} more evidence items</li>
      ) : null}
    </ul>
  );
};

const SummarySection = ({
  label,
  icon: Icon,
  entity,
  papers,
}: {
  label: string;
  icon: ComponentType<{ className?: string }>;
  entity: ComparisonEntitySummary;
  papers: ComparisonPaper[];
}) => (
  <div className="rounded-lg border bg-card p-4 shadow-sm">
    <div className="flex items-center gap-2">
      <span className="rounded-md bg-primary/10 p-2 text-primary">
        <Icon className="h-4 w-4" />
      </span>
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-primary">{label}</p>
        <p className="text-sm text-muted-foreground">Shared vs. unique coverage</p>
      </div>
    </div>
    <div className="mt-4 space-y-3 text-sm">
      <div>
        <p className="font-semibold text-foreground">Shared</p>
        {entity.shared.length ? (
          <ul className="mt-1 list-disc space-y-1 pl-4 text-muted-foreground">
            {entity.shared.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : (
          <p className="mt-1 text-muted-foreground">No shared items captured yet.</p>
        )}
      </div>
      <div>
        <p className="font-semibold text-foreground">What’s unique</p>
        {papers.map((paper) => {
          const entries = entity.unique[paper.id] ?? [];
          return (
            <div key={paper.id} className="mt-2">
              <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{paper.title ?? paper.id}</p>
              {entries.length ? (
                <ul className="mt-1 list-disc space-y-1 pl-4 text-muted-foreground">
                  {entries.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : (
                <p className="mt-1 text-muted-foreground">No unique items recorded.</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  </div>
);

const ComparisonTable = ({
  rows,
  papers,
}: {
  rows: ComparisonRow[];
  papers: ComparisonPaper[];
}) => {
  if (!rows.length) {
    return (
      <div className="rounded-lg border border-dashed border-muted-foreground/30 bg-muted/20 p-6 text-sm text-muted-foreground">
        No extracted results to compare yet. Trigger extraction to populate the matrix.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border bg-card shadow-sm">
      <table className="min-w-full divide-y divide-border text-sm">
        <thead className="bg-muted/40 text-left text-xs uppercase text-muted-foreground">
          <tr>
            <th scope="col" className="px-4 py-3 font-semibold">Dataset</th>
            <th scope="col" className="px-4 py-3 font-semibold">Metric</th>
            <th scope="col" className="px-4 py-3 font-semibold">Method</th>
            <th scope="col" className="px-4 py-3 font-semibold">Task</th>
            <th scope="col" className="px-4 py-3 font-semibold">Split</th>
            {papers.map((paper) => (
              <th key={paper.id} scope="col" className="px-4 py-3 font-semibold">
                {paper.title ?? paper.id}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-border text-sm">
          {rows.map((row, index) => (
            <tr key={index}>
              <td className="px-4 py-3 align-top font-medium text-foreground">{row.dataset_name ?? "—"}</td>
              <td className="px-4 py-3 align-top text-muted-foreground">
                <div className="space-y-1">
                  <p>{row.metric_name ?? "—"}</p>
                  {row.metric_unit ? <p className="text-xs">Unit: {row.metric_unit}</p> : null}
                </div>
              </td>
              <td className="px-4 py-3 align-top text-muted-foreground">{row.method_name ?? "—"}</td>
              <td className="px-4 py-3 align-top text-muted-foreground">{row.task_name ?? "—"}</td>
              <td className="px-4 py-3 align-top text-muted-foreground">{row.split ?? "—"}</td>
              {papers.map((paper) => {
                const cell = row.papers[paper.id];
                return (
                  <td key={paper.id} className="px-4 py-3 align-top">
                    {cell ? (
                      <div className="space-y-2">
                        <div>
                          <p className="font-semibold text-foreground">{formatValue(cell)}</p>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                            {cell.is_sota ? (
                              <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 font-medium text-emerald-700">
                                <CheckCircle2 className="h-3 w-3" /> SOTA
                              </span>
                            ) : null}
                            {typeof cell.confidence === "number" ? (
                              <span className="rounded-full bg-muted px-2 py-0.5">Confidence {(cell.confidence * 100).toFixed(0)}%</span>
                            ) : null}
                          </div>
                        </div>
                        <EvidenceList evidence={cell.evidence} />
                      </div>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default function ComparePage() {
  const [papers, setPapers] = useState<PaperListItem[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<PaperComparisonResponse | null>(null);

  const fetchPapers = useCallback(async () => {
    try {
      const response = await axios.get<PaperListItem[]>(`${API_BASE_URL}/api/papers`, {
        params: { limit: 200 },
      });
      const sorted = [...response.data].sort((a, b) => {
        const titleA = a.title?.toLowerCase() ?? "";
        const titleB = b.title?.toLowerCase() ?? "";
        if (titleA === titleB) {
          return 0;
        }
        return titleA < titleB ? -1 : 1;
      });
      setPapers(sorted);
      setError(null);
    } catch (err) {
      setError(getErrorMessage(err, "Failed to load papers for comparison."));
    }
  }, []);

  useEffect(() => {
    void fetchPapers();
  }, [fetchPapers]);

  const refreshComparison = useCallback(
    async (paperIds: string[]) => {
      if (paperIds.length < 2) {
        setComparison(null);
        return;
      }

      setIsRefreshing(true);
      setError(null);
      try {
        const response = await axios.get<PaperComparisonResponse>(`${API_BASE_URL}/api/compare`, {
          params: { papers: paperIds.join(",") },
        });
        setComparison(response.data);
      } catch (err) {
        setError(getErrorMessage(err, "Unable to compute comparison. Please try again."));
      } finally {
        setIsRefreshing(false);
      }
    },
    []
  );

  const handleSelectionChange = useCallback(
    (event: ChangeEvent<HTMLSelectElement>) => {
      const options = Array.from(event.target.selectedOptions).map((option) => option.value);
      setSelected(options);
      void refreshComparison(options);
    },
    [refreshComparison]
  );

  const handleRefreshClick = useCallback(() => {
    if (selected.length >= 2) {
      setIsLoading(true);
      void refreshComparison(selected).finally(() => setIsLoading(false));
    }
  }, [refreshComparison, selected]);

  const comparisonPapers = useMemo(() => comparison?.papers ?? [], [comparison]);

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-primary">Compare Papers</p>
        <h1 className="text-2xl font-semibold text-foreground">Spot shared findings and fresh contributions</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Select at least two papers—whether they come from machine learning, biology, chemistry, physics, or any other
          discipline—to see overlapping methods, datasets, and metrics. The comparison matrix simply reflects the structured
          results we store, so it stays domain agnostic while still surfacing supporting evidence snippets.
        </p>
      </div>

      <div className="flex flex-col gap-3 rounded-lg border bg-card p-4 shadow-sm md:flex-row md:items-center md:justify-between">
        <div className="flex flex-col gap-2">
          <label htmlFor="paper-selection" className="text-sm font-medium text-foreground">
            Choose papers
          </label>
          <select
            id="paper-selection"
            multiple
            value={selected}
            onChange={handleSelectionChange}
            className="h-40 min-w-[16rem] rounded-md border border-border bg-background p-2 text-sm shadow-inner focus:border-primary focus:outline-none"
          >
            {papers.map((paper) => (
              <option key={paper.id} value={paper.id}>
                {formatPaperLabel(paper)}
              </option>
            ))}
          </select>
          <p className="text-xs text-muted-foreground">Hold Ctrl/Cmd to pick multiple entries.</p>
        </div>
        <div className="flex flex-col items-start gap-2 md:items-end">
          <button
            type="button"
            onClick={handleRefreshClick}
            disabled={selected.length < 2 || isLoading}
            className="inline-flex items-center gap-2 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground shadow transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:bg-muted"
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowRightLeft className="h-4 w-4" />}
            Run comparison
          </button>
          {isRefreshing ? (
            <p className="flex items-center gap-2 text-xs text-muted-foreground">
              <RefreshCw className="h-3 w-3 animate-spin" /> Updating comparison
            </p>
          ) : null}
        </div>
      </div>

      {error ? (
        <div className="flex items-start gap-3 rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          <AlertCircle className="mt-0.5 h-4 w-4" />
          <p>{error}</p>
        </div>
      ) : null}

      {comparison ? (
        <>
          <section className="grid gap-4 lg:grid-cols-2">
            <SummarySection label="Methods" icon={Table2} entity={comparison.summary.methods} papers={comparisonPapers} />
            <SummarySection label="Datasets" icon={Table2} entity={comparison.summary.datasets} papers={comparisonPapers} />
            <SummarySection label="Metrics" icon={Table2} entity={comparison.summary.metrics} papers={comparisonPapers} />
            <SummarySection label="Tasks" icon={Table2} entity={comparison.summary.tasks} papers={comparisonPapers} />
          </section>

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-foreground">Extracted result matrix</h2>
                <p className="text-sm text-muted-foreground">
                  Compare metric values across the selected papers. Evidence snippets show where each claim was found.
                </p>
              </div>
            </div>
            <ComparisonTable rows={comparison.matrix} papers={comparisonPapers} />
          </section>
        </>
      ) : (
        <div className="rounded-lg border border-dashed border-muted-foreground/40 bg-muted/20 p-6 text-sm text-muted-foreground">
          Select at least two papers to generate a comparison. Results and summaries will appear here.
        </div>
      )}
    </div>
  );
}
