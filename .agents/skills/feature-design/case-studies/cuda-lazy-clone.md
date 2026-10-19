# Case study: CUDA copy-on-write with streams and graphs (October 2026)

Materializing a lazily cloned (`_lazy_clone()`, copy-on-write) CUDA tensor
copied with a synchronous `cudaMemcpy` on the legacy default stream. The copy
wasn't ordered with non-blocking streams and wasn't captured into CUDA graphs,
so replays read stale data (#200155). Six designs were written. Five were
rejected and #200800 was accepted. The principle numbers below refer to
[../SKILL.md](../SKILL.md).

## The designs

| PR | Idea | Why it lost |
|---|---|---|
| #200775 | Make the copy async on the current stream and document a rule: anyone doing an async read must hold another lazy clone until the read finishes. | Simple and fast, but a user who forgets the rule gets silently wrong results (principle 1). |
| #200106 | Keep one shared stream per COW context. Copies must happen on it; cross-stream steals are fenced on it; copies are banned during capture, but lazy clones of graph inputs during capture stay lazy. | Adversarial review kept finding holes: skipping the fence during capture dropped ordering between streams in the same capture (principle 6); moving the shared stream lost dependencies (principle 11); an eager-copy read on a side stream wasn't covered by the fence. The authors stopped patching it. |
| #200803 → #200804 → #200805 | Never steal on CUDA. Copy into an allocation on the allocation stream and keep the old data alive until the storage dies. Honor `record_stream`. Add `_lazy_clone_release(t, event)` to opt back into stealing once every reference ever created is released. | Correct but conservative. Superseded by a version of the same idea with a cheaper default and a safer opt-in. |
| #200800 | Same allocation-stream and keep-alive structure. A plain deletion relies on the existing allocator contract, so stealing by default fences on the allocation stream. A materialized storage keeps the old data until the user calls `_cow_record_done(t)`, which records an event; any later access to the data cancels that. Lazy clones of memory allocated during the same capture are allowed; lazy clones of graph inputs during capture raise an error. | Accepted. |

## How the principles showed up

- **Ranking failure modes (1)** decided between #200775 and the rest. Keeping
  old data alive (more memory) was preferred both to silent corruption and to
  stalls on unrelated work.
- **Building on existing contracts (3).** If the refcount reaches one because
  the other references were freed, a correct program has already synced their
  uses back to the allocation stream. That is the allocator's existing rule,
  so a steal only needs to fence on that stream.
- **Hidden transitions (4, 5).** The new case was that writing to a lazy clone
  implicitly releases the shared original. It doesn't look like a free, so
  users won't sync for it. #200800 handles it with an explicit declaration
  instead of a new implicit rule.
- **Requiring all pending async reads to be joined before `_lazy_clone()`** was
  considered and rejected: it makes something that looks like a read require
  the synchronization of a write (4).
- **Tracking readers via `data_ptr()` streams** was considered and rejected:
  the stream isn't reliable and only a stream (not an event) can be stored, so
  syncs would be late (7). #200800 still hooks data access, but only to cancel
  a completion, which is the safe direction.
- **The opt-in can't lie (2).** Cancelling a completion on later access means
  a careless caller gets an extra copy, not a race.
- **Dropping a special case (10).** #200106's motivating optimization, keeping
  in-capture clones of static inputs lazy, depended on knowing the input
  outlives the graph, which can't be known inside a capture. #200800 dropped
  it.
- **A deliberately broken principle.** #200800 still ignores `record_stream`
  on COW tensors, which breaks principles 2 and 3 for code that relies on
  `record_stream` instead of syncing back. The maintainers knowingly deferred
  it to a follow-up. That is the expected way to break a principle: on purpose,
  and stated.
