import functools as ft
from typing import Dict, List, NamedTuple, Optional, Union, Sequence, Iterable
import itertools
import math


class MergerRule(NamedTuple):
    into: List[int]
    ordering: List[int]


class Merger(NamedTuple):
    mergerules: Dict[int, MergerRule]
    completions: Dict[int, List[int]]
    mergelens: List[int]


class Root(NamedTuple):
    id: int
    head: int


class MergeTree(NamedTuple):
    head: int
    children: Union[List["MergeTree"], Root]
    isnew: bool


def combinations(seq: Sequence[int]):
    return itertools.chain.from_iterable(
        itertools.combinations(seq, i) for i in range(len(seq))
    )


def muticombinations(
    iterable: Sequence[int], lens: Optional[List[int]] = None, min_groups: int = 0
):
    """Compute all possible combinations of a `set` with multiple combinations
    allowed.

    That is, for the set [1,2,3,4], we allow [(1,2),(3,4)]. An optional
    arguments `lens` can be passed to specify the maximum combination length. If
    `lens` is [2,3], then only combinations of length 2 and 3 will be returned.
    """
    combs = [[[]]]
    s = set(iterable)

    for com in combinations(iterable):
        com = list(com)
        if lens and not (len(com) in lens):
            continue

        remaining = list(s.symmetric_difference(com))
        if min_groups < 2:
            new_comb = [com, *[[e] for e in remaining]]
            combs.append(new_comb)

        sub_combs = muticombinations(remaining, lens=lens, min_groups=0)

        for sub_comb in sub_combs:
            n = sum(map(lambda c: len(c) > 1, sub_comb))
            if n + 1 > min_groups:
                combs.append([[c for c in com], *sub_comb])

    return set(map(lambda el: sorted(el, key=lambda a: min(a)), combs))


def make_muticombinations(n: int, merger: Merger):
    return muticombinations(range(n), lens=merger.mergelens, min_groups=2)


def prod(iterable: Iterable):
    return ft.reduce(lambda x, y: x * y, iterable)


def canmerge(merger: Merger, heads: Sequence[int]):
    return merger.mergerules.get(prod(heads)) is not None


def count_new(trees: Sequence[MergeTree]):
    return len(list(itertools.filterfalse(lambda t: t.isnew, trees)))


def heads(trees: Sequence[MergeTree]):
    return map(lambda t: t.head, trees)


def all_have_n_new(groups, n):
    """Determine if every group in groups has at least `n` new trees  (ignoring
    groups with only one tree.)
    """
    return all(
        filter(lambda t: count_new(t) >= n, filter(lambda g: len(g) > 1, groups))
    )


def getperm(seq1: Sequence, seq2: Sequence):
    s1 = sorted(enumerate(seq1), key=lambda t: t[1])
    s2 = sorted(enumerate(seq2), key=lambda t: t[1])
    return [t[1] for t in sorted([(t1[0], t2[0]) for t1, t2 in zip(s1, s2)])]


def mergeone(merged: List[List[MergeTree]], trees: List[MergeTree], merger: Merger):
    """
    Given a set of tree, find all possible ways to merge them to form a new
    tree, add the results to the `merged` array and return `true` to indicate
    the grouping was successful. If the merge failed, return false.
    """
    if len(trees) == 1:
        pass

    hs = tuple(heads(trees))
    if not canmerge(merger, hs):
        return None

    rule = merger.mergerules[prod(hs)]
    perm = getperm(rule.ordering, hs)
    sub = [trees[p] for p in perm]
    merged.append([MergeTree(h, sub, True) for h in rule.into])
    return merged


def mergeall(trees: List[MergeTree], merger: Merger):
    """Merge em"""
    ntrees = len(trees)
    if ntrees == 1:
        return [trees]

    new_trees: List[List[MergeTree]] = []

    for comb in make_muticombinations(ntrees, merger):
        groups = [[trees[i] for i in c] for c in comb]
        if all_have_n_new(groups, 1):
            good = True
            m = len(groups)
            cnt = 1
            while good and cnt <= m:
                group = groups.pop()
                good = mergeone(groups, group, merger)
                cnt += 1

            if good:
                new_trees = unfold_into(new_trees, groups)

    return new_trees


def unfold_into(newtrees: List[List[MergeTree]], groups: List[List[MergeTree]]):
    """Flatten out the trees and add them into the array. Currently, the trees
    have entries like [...([T₁,...,Tₙ])...] where the different T's come from
    different merges. We want to flatten this out so we get all possible
    mergings
    """
    m = prod(map(len, groups))
    ngroups = len(groups)
    for i in range(m):
        freq = 1
        tree: List[MergeTree] = []
        for j in range(ngroups):
            length = len(groups[j])
            freq *= length
            tree[j] = groups[j][math.floor(freq * (i - 1) / m) % length + 1]
        newtrees.append(tree)

    return newtrees


def build_trees(roots: List[Root], merger: Merger):
    """Given a set of roots, turn them into `MTree`s and continue to merge the
    trees into fewer trees using the `mergerules`, which is a dictionary maping
    tuples of heads to a new head: mergerules = Dict([(A,B,C) -> (D,E), ...]).
    After each merging, check if there is a way to `complete` the tree by
    combining all subtrees into a single tree using the `completions` array
    which consists of a vector of tuples specifying which sets of trees can be
    completed: completions = [(A,B,C,D), (A,B,C,E), ...]  Then return all the
    completed trees.
    """
    completed: List[List[MergeTree]] = [[]]
    incomplete: List[List[MergeTree]] = [[]]
    incomplete.append([MergeTree(root.head, root, True) for root in roots])

    while len(incomplete) > 0:
        trees = incomplete.pop()
        hs = tuple(heads(trees))
        key = prod(hs)
        if count_new(trees) >= 2 and merger.completions.get(key):
            perm = getperm(merger.completions[key], hs)
            trees = [trees[p] for p in perm]
            completed.append(trees)

        merged = mergeall(trees, merger)
        if len(merged) > 0:
            incomplete.extend(merged)

    return completed
