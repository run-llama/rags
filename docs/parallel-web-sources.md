# Prepare web sources for a RAG agent

Use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
to find web sources and save fetched text as a local dataset for RAGs. The
anonymous endpoint offers free search and page fetching without a Parallel
account or API key. Search uses Fast mode; free usage has rate limits.

This workflow prepares a source snapshot before indexing. RAGs continues to use
its existing file loader, LLM configuration and credentials. It does not connect
the generated agent to MCP or add live search to the chat.

## 1. Connect and find sources

You need Bash, Python 3, and Node.js 22.19 or newer with `npx`. Run these commands
in a separate working directory where you want to keep the source files. The
[MCP Inspector CLI](https://github.com/modelcontextprotocol/inspector/tree/main/clients/cli)
runs outside RAGs' Python environment; no RAGs dependencies need to change.

Define a helper for the Streamable HTTP connection and check the available tools:

```bash
parallel_mcp() {
  npx --yes @modelcontextprotocol/inspector@2.11.0 --cli \
    --transport http --server-url https://search.parallel.ai/mcp \
    --header 'User-Agent: rags-source-preparation/0.0.5' \
    --stored-auth-only --format json "$@"
}
parallel_mcp --method tools/list
```

The tool list should include `web_search` and `web_fetch`. Search for a topic you
want your RAG agent to answer questions about. For example:

```bash
research_session=$(python -c 'import uuid; print(uuid.uuid4())')
parallel_mcp --method tools/call --tool-name web_search \
  --tool-arg 'objective=Find official Python documentation explaining how to create and activate virtual environments.' \
  --tool-arg 'search_queries=["Python venv create virtual environment", "Python venv activate environment"]' \
  --tool-arg "session_id=$research_session" > search.json
python -m json.tool search.json
```

Inspect the returned URLs and excerpts before choosing your sources. Keep the
same `research_session` for related search and fetch calls. For your own topic,
replace the objective and queries, then use the URLs you selected in the next step.

## 2. Fetch and save the dataset

For the Python example, fetch the official `venv` documentation. Fetch returns
focused excerpts by default, so describe the information your dataset needs:

```bash
parallel_mcp --method tools/call --tool-name web_fetch \
  --tool-arg 'urls=["https://docs.python.org/3/library/venv.html"]' \
  --tool-arg 'objective=Find commands for creating a virtual environment and activating it on Linux.' \
  --tool-arg 'search_queries=["Python venv create virtual environment", "Python venv activate environment"]' \
  --tool-arg "session_id=$research_session" > fetch.json
```

Continue only if the command succeeds. If it reports a quota or server error,
wait as directed by the service before retrying. Do not load an error response
as source material.

Convert the successful result's excerpts into one UTF-8 file, retaining each
page's title and source URL:

```bash
python - <<'PY'
import json
from pathlib import Path

result = json.loads(Path("fetch.json").read_text(encoding="utf-8"))["result"]
if result.get("isError"):
    raise SystemExit("Fetch failed; inspect fetch.json before continuing.")
payload = result["structuredContent"]
if payload["errors"] or not payload["results"]:
    raise SystemExit("Some sources could not be fetched; inspect fetch.json.")
sections = []
for page in payload["results"]:
    excerpts = "\n\n".join(page["excerpts"]).strip()
    if not excerpts:
        raise SystemExit("No excerpts for " + page["url"])
    title = page.get("title") or page["url"]
    sections.append("# " + title + "\nSource: " + page["url"] + "\n\n" + excerpts)
text = "\n\n".join(sections)
output = Path("parallel-sources.txt")
output.write_text(text + "\n", encoding="utf-8")
print(output.resolve())
PY
```

Open `parallel-sources.txt` and check that it contains useful source text and
URLs. The file contains the returned excerpts, which may omit parts of a page.
Choose a narrower objective or different sources if it lacks the information
you need.

## 3. Load it into RAGs

Complete the [RAGs installation and setup](../README.md#installation-and-setup)
and start the app. In the **Home** page, give the builder the absolute path
printed above, for example:

> Load the local file /absolute/path/to/parallel-sources.txt. Build an agent that
> answers questions about creating and activating Python virtual environments.

The file must be on the machine running Streamlit. If RAGs runs on another
machine, copy the file there and use its path on that machine.

Review the generated parameters in **RAG Config**, then open **Generated RAG
Agent** and ask:

> Which command creates a virtual environment, and how do I activate it on Linux?

The agent answers over the saved dataset. To refresh the information, fetch new
source text and build a new agent from the updated file.
