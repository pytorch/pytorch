using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
namespace toc
{
    public class Builder
    {
        public static string getAnchorId(string title)
        {
            StringBuilder sb = new StringBuilder();
            title = title.Trim().ToLower();

            foreach (var ch in title)
            {
                if (ch >= 'a' && ch <= 'z' || ch >= '0' && ch <= '9'
                    || ch == '-'|| ch =='_')
                    sb.Append(ch);
                else if (ch == ' ' )
                    sb.Append('-');
            }
            return sb.ToString();
        }

        public class Node
        {
            public List<string> fileNamePrefix = new List<string>();
            public string title;
            public string shortTitle;
            public string fileID;
            public List<string> sections = new List<string>();
            public List<string> sectionShortTitles = new List<string>();
            public List<Node> children = new List<Node>();
        }

        public static void buildTOC(StringBuilder sb, Node n)
        {
            sb.AppendFormat("<li data-link=\"{0}\"><span>{1}</span>\n", n.fileID, n.shortTitle);
            if (n.children.Count != 0)
            {
                sb.AppendLine("<ul class=\"toc_list\">");
                foreach(var c in n.children)
                    buildTOC(sb, c);
                sb.AppendLine("</ul>");
            }
            else if (n.sections.Count != 0)
            {
                sb.AppendLine("<ul class=\"toc_list\">");
                for (int i = 0; i < n.sections.Count; i++)
                {
                    var s = n.sections[i];
                    var shortTitle = n.sectionShortTitles[i];
                    sb.AppendFormat("<li data-link=\"{0}#{1}\"><span>{2}</span></li>\n", n.fileID, getAnchorId(s), shortTitle);
                }
                sb.AppendLine("</ul>");
            }
            sb.AppendLine("</li>");
        }
        public static string buildTOC(Node n)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(@"<ul class=""toc_root_list"">");
            buildTOC(sb, n);
            sb.Append(@"</ul>");
            return sb.ToString();
        }

        public static bool isChild(Node parent, Node child)
        {
            if (parent.fileNamePrefix.Count < child.fileNamePrefix.Count)
            {
                bool equal = true;
                for (int k = 0; k < parent.fileNamePrefix.Count; k++)
                {
                    if (parent.fileNamePrefix[k] != child.fileNamePrefix[k])
                    {
                        equal = false;
                        break;
                    }
                }
                return equal;
            }
            return false;
        }

        public static string getNextNonEmptyLine(string[] lines, int i)
        {
            i++;
            while (i < lines.Length)
            {
                if (lines[i].Trim().Length != 0)
                    return lines[i];
                i++;
            }
            return "";
        }
        const string shortTitlePrefix = "[//]: # (ShortTitle: ";

        public static string maybeGetShortTitleImpl(string originalTitle, string[] lines, int line)
        {
            string nextLine = getNextNonEmptyLine(lines, line);
            if (nextLine.StartsWith(shortTitlePrefix))
            {
                return nextLine.Substring(shortTitlePrefix.Length, nextLine.Length - shortTitlePrefix.Length - 1).Trim();
            }
            return originalTitle;
        }

        public static string escapeString(string input)
        {
            StringBuilder sb = new StringBuilder();
            foreach (var ch in input)
            {
                if (ch == '<')
                    sb.Append("&lt;");
                else if (ch == '>')
                    sb.Append("&gt;");
                else
                    sb.Append(ch);
            }
            return sb.ToString();
        }
        public static string maybeGetShortTitle(string originalTitle, string[] lines, int line)
        {
            string title = maybeGetShortTitleImpl(originalTitle, lines, line);
            return escapeString(title);
        }
        public static string Run(string path)
        {
            StringBuilder outputSB = new StringBuilder();
            outputSB.AppendFormat("Building table of contents from {0}...\n", path);
            var files = System.IO.Directory.EnumerateFiles(path, "*.md").OrderBy(f => System.IO.Path.GetFileName(f));
            List<Node> nodes = new List<Node>();
            foreach (var f in files)
            {
                var content = File.ReadAllLines(f);
                Node node = new Node();
                node.fileID = Path.GetFileNameWithoutExtension(f);
                outputSB.AppendFormat("  {0}.md\n", node.fileID);
                bool mainTitleFound = false;
                for (int i = 1; i < content.Length; i++)
                {
                    if (content[i].StartsWith("==="))
                    {
                        mainTitleFound = true;
                        node.title = content[i-1];
                        node.shortTitle = maybeGetShortTitle(node.title, content, i);
                    }
                    if (content[i].StartsWith("---"))
                    {
                        if (!mainTitleFound) continue;
                        node.sections.Add(content[i-1]);
                        node.sectionShortTitles.Add(maybeGetShortTitle(content[i - 1], content, i));
                    }
                    if (content[i].StartsWith("#") && !content[i].StartsWith("##") && node.title == null)
                    {
                        mainTitleFound = true;
                        node.title = content[i].Substring(1, content[i].Length - 1).Trim();
                        node.shortTitle = maybeGetShortTitle(node.title, content, i);
                    }
                    if (content[i].StartsWith("##") && !content[i].StartsWith("###"))
                    {
                        if (!mainTitleFound) continue;
                        var sectionStr = content[i].Substring(2, content[i].Length - 2).Trim();
                        node.sections.Add(sectionStr);
                        node.sectionShortTitles.Add(maybeGetShortTitle(sectionStr, content, i));
                    }
                    if (content[i].StartsWith("permalink:"))
                    {
                        var prefixLength = ("permalink:").Length;
                        var permaPath = content[i].Substring(prefixLength, content[i].Length - prefixLength).Trim();
                        node.fileID = Path.GetFileName(permaPath);
                    }
                }
                if (node.title == null)
                {
                    outputSB.AppendFormat("Error: {0} does not define a title.", f);
                    node.title = "Untitiled";
                }
                var titleSecs = Path.GetFileName(f).Split('-');
                foreach (var s in titleSecs)
                {
                    if (s.Length == 2 && s[1]>='0' && s[1] <= '9')
                    {
                        node.fileNamePrefix.Add(s);
                    }
                    else
                    {
                        break;
                    }
                }
                // Find parent node.
                Node parent=null;
                for (int l = nodes.Count-1; l>=0; l--)
                {
                    var n = nodes[l];
                    if (isChild(n, node))
                    {
                       parent = n;
                       break;
                    }
                }
                if (parent != null)
                    parent.children.Add(node);
                else
                {
                    // find child
                    foreach (var other in nodes)
                    {
                        if (isChild(node, other))
                        {
                            node.children.Add(other);
                        }
                    }
                    foreach (var c in node.children)
                    {
                        nodes.Remove(c);

                    }
                    nodes.Add(node);
                }
            }
            var root = nodes.Find(x=>x.fileID=="index");
            if (root != null)
            {
                var html = buildTOC(root);
                var outPath = Path.Combine(path, "toc.html");
                File.WriteAllText(outPath, html);
                outputSB.AppendFormat("Output written to: {0}\n", outPath);
            }
            return outputSB.ToString();
        }
    }
}
