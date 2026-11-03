"""Compare selected original and consolidated kernel ASTs after static specialization.

This is a source-level check, not a substitute for compilation or NPU validation.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / 'reference'
dst = ROOT / 'dsa'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=ROOT / 'validation/ast-equivalence.json')
args = parser.parse_args()
records = []
renames = {
    'query_l1': 'q',
    'prob_l1': 'p',
    'prob_flat_l1': 'pflat',
    'keys_l1': 'k0',
    'scores_ub': 'score',
    'prob_padded_ub': 'pub',
    'shared_bytes': 'pv_storage',
    'pv_ub': 'pv',
    'numerator_ub': 'acc',
    'output_bf16': 'out',
    'need_safe_gather': 'any_duplicate',
    'running_max': 'm',
    'running_sum': 'l',
    'rescale': 'alpha',
    'score_l0c': 'sc',
    'pv_l0c': 'pc',
    'gather_key_even': 'gk0',
    'gather_key_odd': 'gk1',
    'gather_rope_even': 'gr0',
    'gather_rope_odd': 'gr1',
    'scale_old': 'a',
}


class Specialize(ast.NodeTransformer):

    def __init__(self, env, renamed=False):
        self.env = dict(env)
        self.renamed = renamed

    def visit_Name(self, node):
        name = renames.get(node.id, node.id) if self.renamed else node.id
        if node.id in self.env and isinstance(node.ctx, ast.Load): return ast.Constant(self.env[node.id])
        if name in ('ext', 'dsa_ext'): name = 'ops'
        return ast.Name(name, node.ctx)

    def visit_AnnAssign(self, node):
        value = self.visit(node.value)
        if isinstance(value, ast.Constant):
            self.env[node.target.id] = value.value
            return None
        node.value = value
        return node

    def visit_If(self, node):
        test = self.visit(node.test)
        if isinstance(test, ast.Constant):
            out = []
            for n in node.body if test.value else node.orelse:
                x = self.visit(n)
                if isinstance(x, list): out.extend(x)
                elif x is not None: out.append(x)
            return out
        node.test = test
        return self.generic_visit(node)

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp)) and not any(
                isinstance(x, (ast.Name, ast.Call, ast.Attribute)) for x in ast.walk(node)):
            try:
                return ast.Constant(
                    eval(compile(ast.fix_missing_locations(ast.Expression(node)), '<static>', 'eval'),
                         {'__builtins__': {}}))
            except Exception:
                pass
        return node


def body(path, env, renamed=False):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name.endswith(('attention', 'kernel')))
    fn = Specialize(env, renamed).visit(fn)
    return ast.Module(body=fn.body, type_ignores=[])


cases = [('fastgather', 128, 128, 256, True), ('wide_fastgather', 16, 128, 512, False),
         ('wide_fastgather', 32, 128, 512, False), ('wide_fastgather', 64, 128, 512, False),
         ('bn256', 16, 256, 512, False)]
for old, bm, bn, pv, share in cases:
    env = dict(BM=bm, BN=bn, PV_COLS=pv, SHARE_SCORE=share)
    a = body(src / f'dsa_kernel_{old}.py', env)
    b = body(dst / 'workspace.py', env, True)
    sa, sb = ast.dump(a), ast.dump(b)
    if sa != sb:
        import difflib
        oldtext = ast.unparse(a)
        newtext = ast.unparse(b)
        print('\n'.join(
            difflib.unified_diff(oldtext.splitlines(), newtext.splitlines(), fromfile=old, tofile='new')))
        raise SystemExit(f'MISMATCH {old} BM={bm}')
    records.append(
        dict(original=f'dsa_kernel_{old}.py',
             refactored='workspace.py',
             bm=bm,
             bn=bn,
             pv_cols=pv,
             share_score=share,
             equivalent=True))
    print('AST EQUIVALENT', old, bm, bn, pv, share)
# Ragged and merge are exactly the same device AST, excluding import namespaces,
# function name and comments. Their kernel body is not restructured.
for old, new, fn in [('dsa_kernel_ragged.py', 'ragged.py', 'ragged'), ('merge.py', 'merge.py', 'merge')]:
    oa = ast.parse((src / old).read_text())
    na = ast.parse((dst / new).read_text())
    a = next(n for n in oa.body if isinstance(n, ast.FunctionDef) and n.decorator_list)
    b = next(n for n in na.body if isinstance(n, ast.FunctionDef) and n.decorator_list)
    a = Specialize({}).visit(a)
    b = Specialize({}).visit(b)
    assert ast.dump(ast.Module(a.body, [])) == ast.dump(ast.Module(b.body, [])), fn
    records.append(dict(original=old, refactored=new, equivalent=True))
    print('AST EQUIVALENT', fn)

record = {
    "method":
    "Specialize constexpr-only branches/expressions; normalize renamed locals and extension namespace; compare complete AST bodies with statement ordering preserved.",
    "scope":
    "The five workspace specializations cover all 15 selected workspace cases. Ragged decode_b1 and merge bodies are checked separately. This is not an IR, binary, numerical or performance proof.",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "passed": all(row["equivalent"] for row in records),
    "checks": records,
    "source_sha256": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [*src.glob("*.py"), *dst.glob("*.py")]
    },
}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
print("Wrote", args.out)
