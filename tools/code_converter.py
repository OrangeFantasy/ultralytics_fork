from typing import Union
from pathlib import Path

import inspect
import shutil

import libcst as cst
from libcst.metadata import PositionProvider, ParentNodeProvider

class IfWalrusTransformer(cst.CSTTransformer):
    def _extract_namedexpr(self, node: cst.CSTNode):
        assigns = []

        class WalrusReplacer(cst.CSTTransformer):
            def leave_NamedExpr(self, original_node, updated_node):
                assigns.append(
                    cst.SimpleStatementLine(
                        body=[
                            cst.Assign(
                                targets=[cst.AssignTarget(target=updated_node.target)],
                                value=updated_node.value,
                            )
                        ]
                    )
                )
                return updated_node.target

        new_node = node.visit(WalrusReplacer())
        return assigns, new_node

    def leave_If(
        self, original_node: cst.If, updated_node: cst.If
    ) -> Union[
        cst.BaseStatement,
        cst.FlattenSentinel[cst.BaseStatement],
        cst.RemovalSentinel,
    ]:
        assigns, new_test = self._extract_namedexpr(updated_node.test)
        if not assigns:
            return updated_node
        new_if = updated_node.with_changes(test=new_test)
        return cst.FlattenSentinel(assigns + [new_if])

class TypeHintTransformer(cst.CSTTransformer):
    def __init__(self):
        self.used = set()
        self._in_annotation = 0

    def _flatten_pep604(self, node: cst.BaseExpression) -> list[cst.BaseExpression]:
        if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.BitOr):
            return self._flatten_pep604(node.left) + self._flatten_pep604(node.right)
        return [node]

    def leave_Subscript(
        self, original_node: cst.Subscript, updated_node: cst.Subscript
    ) -> cst.BaseExpression:
        if isinstance(updated_node.value, cst.Name):
            mp = {"list": "List", "dict": "Dict", "set": "Set", "tuple": "Tuple"}
            if updated_node.value.value in mp:
                name = mp[updated_node.value.value]
                self.used.add(name)
                return updated_node.with_changes(value=cst.Name(name))
        return updated_node

    def leave_BinaryOperation(
        self, original_node: cst.BinaryOperation, updated_node: cst.BinaryOperation
    ) -> cst.BaseExpression:
        if self._in_annotation == 0:
            return updated_node
        if not isinstance(updated_node.operator, cst.BitOr):
            return updated_node

        items = self._flatten_pep604(updated_node)
        has_none = any(isinstance(x, cst.Name) and x.value == "None" for x in items)
        items = [x for x in items if not (isinstance(x, cst.Name) and x.value == "None")]

        if len(items) == 1:
            inner = items[0]
        else:
            self.used.add("Union")
            inner = cst.Subscript(
                value=cst.Name("Union"),
                slice=[cst.SubscriptElement(slice=cst.Index(x)) for x in items],
            )

        if has_none:
            self.used.add("Optional")
            inner = cst.Subscript(
                value=cst.Name("Optional"),
                slice=[cst.SubscriptElement(slice=cst.Index(inner))],
            )

        return inner

    def visit_Annotation(self, node: cst.Annotation):
        self._in_annotation += 1

    def leave_Annotation(
        self, original_node: cst.Annotation, updated_node: cst.Annotation
    ) -> cst.Annotation:
        self._in_annotation -= 1

        ann = updated_node.annotation
        items = self._flatten_pep604(ann)

        if len(items) <= 1:
            return updated_node

        has_none = any(isinstance(x, cst.Name) and x.value == "None" for x in items)
        items = [x for x in items if not (isinstance(x, cst.Name) and x.value == "None")]

        if len(items) == 1:
            inner = items[0]
        else:
            self.used.add("Union")
            inner = cst.Subscript(
                value=cst.Name("Union"),
                slice=[cst.SubscriptElement(slice=cst.Index(x)) for x in items],
            )

        if has_none:
            self.used.add("Optional")
            inner = cst.Subscript(
                value=cst.Name("Optional"),
                slice=[cst.SubscriptElement(slice=cst.Index(inner))],
            )

        return updated_node.with_changes(annotation=inner)

    def leave_Module(
        self, original_node: cst.Module, updated_node: cst.Module
    ) -> cst.Module:
        if not self.used:
            return updated_node

        stmt = cst.parse_statement(f"from typing import {', '.join(sorted(self.used))}")
        body, i = list(updated_node.body), 0

        # Skip module docstring
        if (
            body 
            and isinstance(body[0], cst.SimpleStatementLine) 
            and isinstance(body[0].body[0], cst.Expr) 
            and isinstance(body[0].body[0].value, cst.SimpleString)
        ):
            i = 1

        # Skip __future__ imports
        while i < len(body):
            s = body[i]
            if (
                isinstance(s, cst.SimpleStatementLine) 
                and isinstance(s.body[0], cst.ImportFrom)
                and isinstance(s.body[0].module, cst.Name) 
                and s.body[0].module.value == "__future__"
            ):
                i += 1
            else:
                break
        
        self.used.clear()
        return updated_node.with_changes(body=body[:i] + [stmt] + body[i:])

class ImportlibMetadataTransformer(cst.CSTTransformer):
    def __init__(self):
        self.aliases = set()

    def leave_Import(
        self, original_node: cst.Import, updated_node: cst.Import
    ) -> Union[cst.BaseSmallStatement, cst.FlattenSentinel[cst.BaseSmallStatement], cst.RemovalSentinel]:
        names, hit = [], False
        for a in updated_node.names:
            n = a.name
            if (
                isinstance(n, cst.Attribute) 
                and isinstance(n.value, cst.Name) 
                and n.value.value == "importlib" 
                and n.attr.value == "metadata"
            ):
                asname = a.asname.name.value if a.asname else "metadata"
                self.aliases.add(asname)
                names.append(cst.ImportAlias(
                    name=cst.Name("importlib_metadata"),
                    asname=cst.AsName(name=cst.Name(asname)),
                ))
                hit = True
            else:
                names.append(a)
        return updated_node.with_changes(names=names) if hit else updated_node

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> Union[cst.BaseSmallStatement, cst.FlattenSentinel[cst.BaseSmallStatement], cst.RemovalSentinel]:
        if isinstance(updated_node.module, cst.Name) and updated_node.module.value == "importlib":
            names, hit = [], False
            for a in updated_node.names:
                if isinstance(a, cst.ImportAlias) and a.name.value == "metadata":
                    names.append(cst.ImportAlias(name=cst.Name("metadata"), asname=a.asname))
                    hit = True
                else:
                    names.append(a)
            if hit:
                return updated_node.with_changes(module=cst.Name("importlib_metadata"), names=names)
        return updated_node

    def leave_Attribute(
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.BaseExpression:
        # Rewrite importlib.metadata.xxx -> <alias>.xxx
        if self.aliases and isinstance(updated_node.value, cst.Attribute) and \
           isinstance(updated_node.value.value, cst.Name) and updated_node.value.value.value == "importlib" and \
           updated_node.value.attr.value == "metadata":
            return updated_node.with_changes(value=cst.Name(next(iter(self.aliases))))
        return updated_node

class CachedPropertyTransformer(cst.CSTTransformer):
    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom,
    ) -> Union[cst.BaseSmallStatement, cst.FlattenSentinel[cst.BaseSmallStatement], cst.RemovalSentinel]:
        if isinstance(updated_node.module, cst.Name) and updated_node.module.value == "functools":
            names, hit = [], False
            for a in updated_node.names:
                if isinstance(a, cst.ImportAlias) and a.name.value == "cached_property":
                    names.append(cst.ImportAlias(name=cst.Name("cached_property"), asname=a.asname))
                    hit = True
                else:
                    names.append(a)
            if hit:
                # from functools import cached_property  ->  from backports.cached_property import cached_property
                return updated_node.with_changes(module=cst.Attribute(value=cst.Name("backports"), attr=cst.Name("cached_property"),), names=names)
        return updated_node

class LruCacheDecoratorTransformer(cst.CSTTransformer):
    def leave_Decorator(self, original_node: cst.Decorator, updated_node: cst.Decorator):
        expr = updated_node.decorator

        # @functools.lru_cache  ->  @functools.lru_cache()
        if (
            isinstance(expr, cst.Attribute)
            and isinstance(expr.value, cst.Name)
            and expr.value.value == "functools"
            and expr.attr.value == "lru_cache"
        ):
            return updated_node.with_changes(decorator=cst.Call(func=expr, args=[]))
        return updated_node

class FstringDebugTransformer(cst.CSTTransformer):
    # def leave_FormattedString(
    #     self, original_node: cst.FormattedString, updated_node: cst.FormattedString
    # ) -> cst.BaseExpression:
    #     parts, changed = [], False
    #     for p in updated_node.parts:
    #         if isinstance(p, cst.FormattedStringExpression) and p.equal and isinstance(p.expression, cst.Name):
    #             var = p.expression.value
    #             parts.append(cst.FormattedStringText(value=f"{var}="))
    #             parts.append(p.with_changes(equal=None))
    #             changed = True
    #         else:
    #             parts.append(p)
    #     return updated_node.with_changes(parts=parts) if changed else updated_node

    def leave_FormattedString(
        self, original_node: cst.FormattedString, updated_node: cst.FormattedString
    ) -> cst.BaseExpression:
        parts, changed = [], False
        for p in updated_node.parts:
            if isinstance(p, cst.FormattedStringExpression) and p.equal:    
                label = cst.Module([]).code_for_node(p.expression)  # Render original expression source as label, e.g. "self.x"       
                parts.append(cst.FormattedStringText(value=f"{label}="))  # Insert plain text: "expr="           
                parts.append(p.with_changes(equal=None))  # Keep original expression, drop debug "="
                changed = True
            else:
                parts.append(p)

        return updated_node.with_changes(parts=parts) if changed else updated_node

try:
    Starred = cst.StarredElement
except AttributeError:
    from libcst._nodes.expression import StarredElement as Starred

class ReturnStarTransformer(cst.CSTTransformer):
    def leave_Return(
        self, original_node: cst.Return, updated_node: cst.Return
    ) -> Union[cst.BaseSmallStatement, cst.FlattenSentinel[cst.BaseSmallStatement], cst.RemovalSentinel]:
        v = updated_node.value
        if isinstance(v, Starred):
            return updated_node.with_changes(
                value=cst.Tuple(
                    elements=[cst.Element(value=v)],  # keep the StarredElement
                    lpar=[cst.LeftParen()],
                    rpar=[cst.RightParen()],
                )
            )
        return updated_node

class Py310_FeatureScanner(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    def __init__(self):
        self.issues = []

    def _report(self, node: cst.CSTNode, msg: str):
        pos = self.get_metadata(PositionProvider, node)
        snippet = cst.Module([]).code_for_node(node).strip()
        self.issues.append(
            f"Line: {pos.start.line}:{pos.start.column}\n"
            f"Code: {snippet}\n"
            f"Message: {msg}\n")

    def visit_Match(self, node: cst.Match):
        self._report(node, "match/case is not supported in Python 3.7")

    def visit_FormattedStringExpression(self, node: cst.FormattedStringExpression):
        if node.equal:
            expr = node.expression
            if isinstance(expr, (cst.Call, cst.Lambda)):
                self._report(
                    node,
                    "f-string debug with call/lambda may change semantics when rewritten for Python 3.7"
                )

    def visit_ImportFrom(self, node: cst.ImportFrom):
        if isinstance(node.module, cst.Name) and node.module.value == "typing":
            for n in node.names:
                if isinstance(n, cst.ImportAlias) and n.name.value in {"Self", "ParamSpec", "TypeAlias", "TypeGuard"}:
                    self._report(node, f"typing.{n.name.value} requires typing_extensions in Python 3.7")

class Py310ToPy37_CodeConverter:
    def __init__(self):
        self.transformers = [
            IfWalrusTransformer(),
            TypeHintTransformer(),
            ImportlibMetadataTransformer(),
            CachedPropertyTransformer(),
            LruCacheDecoratorTransformer(),
            FstringDebugTransformer(),
            ReturnStarTransformer(),
        ]

    def _convert_file(self, path: Path):
        # Read original source
        src = path.read_text(encoding="utf-8")

        # Convert source code (Py310 -> Py37)
        try:
            new_src = self._convert_code(src)
        except RuntimeError as ex:
            raise RuntimeError(
                f"Failed to convert. \n\n"
                f"File: {path} \n" 
                f"Error: {ex}"
            )

        # Write transformed source back
        path.write_text(new_src, encoding="utf-8")

    def _convert_code(self, code: str) -> str:
        module = cst.parse_module(code)
        wrapper = cst.MetadataWrapper(module)

        scanner = Py310_FeatureScanner()
        wrapper.visit(scanner)
        if scanner.issues:
            raise RuntimeError("Found Python 3.10-only features\n" + "\n".join(scanner.issues))

        for t in self.transformers:
            module = module.visit(t)

        return self._debug_codegen(module)

    def _debug_codegen(self, module: cst.Module) -> str:
        try:
            return module.code
        except TypeError:
            print("Codegen error on node:", type(module))
            print(inspect.getsource(type(module)))
            raise

    def convert_tree(self, root: Union[Path, str], sources: list[Union[Path, str]], out_dir: Union[Path, str]):
        # Convert arguments
        if isinstance(root, str):
            root = Path(root)
        if isinstance(out_dir, str):
            out_dir = Path(out_dir)
        for idx in range(len(sources)):
            if isinstance(sources[idx], str):
                sources[idx] = Path(sources[idx])

        # Recreate output workspace
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy source files and directories
        for item in sources:
            src_path = root / item
            dst_path = out_dir / item

            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            elif src_path.is_file():
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
            else:
                raise FileNotFoundError(src_path)

        # Transform all .py files in workspace
        for py in out_dir.rglob("*.py"):
            print(f"Converting {py}...")
            self._convert_file(py)

if __name__ == "__main__":
    import os
    project_dir = os.path.dirname(__file__) + "/.."

    sources = [
        "ultralytics/",

        "tools/fast_math.py",
        "tools/qat/config.py",
        "tools/qat/pipeline.py",
        "tools/qat/Ascend/pipeline.py",

        "main.py",
        "main_qat.py",
    ]
    out_dir = "./.Ascend_QAT"

    converter = Py310ToPy37_CodeConverter()
    converter.convert_tree(project_dir, sources, out_dir)
