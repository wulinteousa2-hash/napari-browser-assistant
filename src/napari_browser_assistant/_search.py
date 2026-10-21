"""A deliberately small, compiled Everything-style filename/path query language."""
import fnmatch
import re


def compile_query(query):
    # Quotes group spaces and literal pipes. An unfinished quote is usable while typing.
    tokens = re.findall(r'!?"[^"]*(?:"|$)|[^\s|]+|\|', query)
    groups, terms = [], []
    for token in tokens:
        if token == "|":
            if terms:
                groups.append(terms)
                terms = []
            continue
        negate = token.startswith("!")
        token = token[1:] if negate else token
        token = token.strip('"').casefold().replace("\\", "/")
        if not token:
            continue
        if token.startswith("ext:"):
            extensions = {"." + e.lstrip(".") for e in token[4:].split(";") if e}
            match = lambda text, ext, extensions=extensions: ext in extensions
        elif "*" in token or "?" in token:
            regex = re.compile(fnmatch.translate(token))
            match = lambda text, ext, regex=regex, token=token: bool(
                regex.fullmatch(text if "/" in token else text.rsplit("/", 1)[-1]))
        else:
            match = lambda text, ext, token=token: token in text
        terms.append((negate, match))
    if terms:
        groups.append(terms)

    def matches(path):
        text = str(path).casefold().replace("\\", "/")
        ext = "." + text.rsplit(".", 1)[-1] if "." in text.rsplit("/", 1)[-1] else ""
        return not groups or any(all(match(text, ext) != negate for negate, match in group)
                                 for group in groups)
    return matches
