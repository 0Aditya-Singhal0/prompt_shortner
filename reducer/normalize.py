import re


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out_lines: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in lines:
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
            out_lines.append(line.rstrip())
            continue

        if in_fence:
            out_lines.append(line.rstrip("\r"))
            continue

        line = line.replace("\t", " ")
        line = re.sub(r" {2,}", " ", line).rstrip()
        out_lines.append(line)

    normalized = "\n".join(out_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
