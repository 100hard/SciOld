"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Database,
  FileText,
  GitCompare,
  LayoutDashboard,
  Settings,
  Share2,
  UploadCloud,
} from "lucide-react";

const navigation = [
  { name: "Overview", href: "/", icon: LayoutDashboard },
  { name: "Graph", href: "/graph", icon: Share2 },
  { name: "Compare", href: "/compare", icon: GitCompare },
  { name: "Datasets", href: "/datasets", icon: Database },
  { name: "Papers", href: "/papers", icon: FileText },
  { name: "Ingestion", href: "/ingestion", icon: UploadCloud },
  { name: "Settings", href: "/settings", icon: Settings },
];

const Sidebar = () => {
  const pathname = usePathname();

  return (
    <aside className="hidden min-h-screen w-64 flex-col border-r bg-card/60 pb-6 pt-8 shadow-sm md:flex">
      <div className="px-6">
        <div className="text-sm font-semibold uppercase tracking-[0.2em] text-primary">SciNets</div>
        <p className="mt-2 text-sm font-medium text-foreground">Knowledge Graph</p>
        <p className="text-xs text-muted-foreground">Manage research entities, pipelines, and integrations.</p>
      </div>
      <nav className="mt-8 flex flex-1 flex-col gap-1 px-3" aria-label="Primary navigation">
        {navigation.map((item) => {
          const Icon = item.icon;
          const isActive =
            pathname === item.href || (item.href !== "/" && pathname?.startsWith(`${item.href}/`));

          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-muted ${
                isActive ? "bg-muted text-foreground shadow-sm" : "text-muted-foreground"
              }`}
              aria-current={isActive ? "page" : undefined}
            >
              <Icon className="h-4 w-4" />
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto px-6">
        <div className="rounded-md border border-dashed border-primary/40 bg-primary/5 p-4 text-xs text-muted-foreground">
          <p className="font-semibold text-foreground">Need more integrations?</p>
          <p>Connect additional sources to keep the knowledge graph fresh.</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
