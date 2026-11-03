"""Compare host validation/allocation behavior with CPU meta tensors only.

The source NPU-device check is deliberately adapted to meta for this test;
this checks Python behavior and host overhead, not NPU execution or timing.
Writes validation/host-api-review-rerun.json without replacing original evidence.
"""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import timeit

import torch


ROOT = Path(__file__).resolve().parents[1]
configs = runpy.run_path(str(ROOT / 'dsa/config.py'))


class Kernel:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            if self.events is not None:
                self.events.append(('launch', self.name, grid, normalize(args), kwargs))
            return self.name
        return launch


def normalize(value):
    if isinstance(value, torch.Tensor):
        return ('tensor', tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, (tuple, list)):
        return tuple(normalize(v) for v in value)
    return value


class TorchSpy:
    def __init__(self, events):
        self.events = events

    def __getattr__(self, name):
        if name not in ('empty', 'empty_like'):
            return getattr(torch, name)
        def allocate(*args, **kwargs):
            result = getattr(torch, name)(*args, **kwargs)
            self.events.append(('allocate', name, normalize(result)))
            return result
        return allocate


def load(path, events=None):
    source = Path(path).read_text().replace('"npu"', '"meta"')
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace = dict(configs, torch=torch if events is None else TorchSpy(events))
    for name in ('ragged_attention', 'workspace_attention', 'merge_partials'):
        namespace[name] = Kernel(name, events)
    exec(compile(tree, path, 'exec'), namespace)
    return namespace


def data_for(shape):
    b, s, h, length, selected = shape
    shapes = [(b, s, h, 512), (b, length, 1, 512), (b, s, h, 64),
              (b, length, 1, 64), (b, s, 1, selected)]
    return tuple(torch.empty(shape, dtype=torch.int32 if i == 4 else torch.bfloat16,
                             device='meta') for i, shape in enumerate(shapes))


before, after = load(str(ROOT / 'validation/host-api-before.txt')), load(str(ROOT / 'dsa/api.py'))
tested = 0
for name, shape in configs['CASE_SHAPES'].items():
    data = data_for(shape)
    for return_kernels in (False, True):
        a, b = [], []
        ra = load(str(ROOT / 'validation/host-api-before.txt'), a)['forward'](*data, return_kernels=return_kernels)
        rb = load(str(ROOT / 'dsa/api.py'), b)['forward'](*data, return_kernels=return_kernels)
        assert a == b, (name, a, b)
        assert normalize(ra) == normalize(rb), name
        tested += 1

data = data_for(configs['CASE_SHAPES']['decode_b4_h32'])
mutations = []
for i, tensor in enumerate(data):
    for shape in (tensor.shape[:-1], tensor.shape[:-1] + (tensor.shape[-1] + 1,),
                  (0,) + tensor.shape[1:]):
        altered = list(data)
        altered[i] = torch.empty(shape, dtype=tensor.dtype, device='meta')
        mutations.append((f'shape-{i}-{tuple(shape)}', altered, {}))
    for kind in ('dtype', 'contiguous', 'device'):
        altered = list(data)
        if kind == 'dtype':
            altered[i] = torch.empty(tensor.shape, dtype=torch.float32, device='meta')
        elif kind == 'contiguous':
            altered[i] = torch.empty(tensor.shape[:-1] + (tensor.shape[-1] * 2,),
                                     dtype=tensor.dtype, device='meta')[..., ::2]
        else:
            altered[i] = torch.empty(tensor.shape, dtype=tensor.dtype, device='cpu')
        mutations.append((f'{kind}-{i}', altered, {}))

for core in (0, -1, True, 1.5, '28'):
    mutations.append((f'core-{core}', data, {'cores': core}))
base = configs['SELECTED']['decode_b4_h32']
for field, values in {'bm': (0, 8, 64, 16.0), 'bn': (0, 64, 127, 128.0),
                      'split': (0, -1, 3, True), 'pv_cols': (0, 64, 256.0)}.items():
    for value in values:
        mutations.append((f'tuning-{field}-{value}', data, {'config': replace(base, **{field: value})}))
mutations.append(('shared-small-pv', data, {'config': replace(base, bn=256, pv_cols=128, split=4, share_score=True)}))
mutations.append(('direct-wrong-bn', data, {'config': replace(base, direct_nz=True)}))
mutations.append(('config-type', data, {'config': {}}))
mutations.append(('unknown-default', data_for((4, 2, 32, 8192, 2048)), {}))

for name, args, kwargs in mutations:
    outcomes = []
    for namespace in (before, after):
        try:
            namespace['forward'](*args, **kwargs)
            outcomes.append('accepted')
        except Exception as exc:
            outcomes.append((type(exc).__name__, str(exc)))
    assert outcomes[0] == outcomes[1], (name, outcomes)
    assert outcomes[0] != 'accepted', (name, outcomes)
    tested += 1


def before_prepare():
    shape = before['_check_inputs'](*data)
    config = before['select'](shape)
    before['_check_tuning'](config, shape, 28)


def after_prepare():
    shape, device = after['_check_inputs'](*data)
    config = after['_DEFAULT_CONFIGS'][shape]
    cores = 28
    if type(cores) is not int or cores <= 0:
        raise ValueError('cores')
    return config, device


functions = {
    'inputs_before': lambda: before['_check_inputs'](*data),
    'inputs_after': lambda: after['_check_inputs'](*data),
    'prepare_before': before_prepare,
    'prepare_after': after_prepare,
    'forward_stub_before': lambda: before['forward'](*data),
    'forward_stub_after': lambda: after['forward'](*data),
}
results = {name: [] for name in functions}
for repeat in range(9):
    names = list(functions)
    offset = repeat % len(names)
    for name in names[offset:] + names[:offset]:
        results[name].append(timeit.timeit(functions[name], number=20000) * 1e6 / 20000)
report = {
    'method': 'Host-only genuine torch meta tensors; only device-type predicate changed to meta; kernel launch stubbed; no NPU allocation or launch',
    'checks': tested,
    'timing_us': {name: {'median': statistics.median(values), 'samples': values}
                  for name, values in results.items()},
    'before_sha256': hashlib.sha256(Path(str(ROOT / 'validation/host-api-before.txt')).read_bytes()).hexdigest(),
    'after_sha256': hashlib.sha256(Path(str(ROOT / 'dsa/api.py')).read_bytes()).hexdigest(),
}
(ROOT / 'validation/host-api-review-rerun.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
