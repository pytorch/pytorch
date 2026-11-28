// What a measured line of torch_members.tsv means (see the head of
// rules/torch.rs): a member's line or a form's, read into a row, a stop or
// a later.

use super::*;

// What a measured line means, a member's or a form's: the columns are
// member, form, kind, cpu, cuda, scpu, scuda, out, P, W, K, D, S, R, A, L,
// O, X, domain (the head of the file says what each is). A line that is none
// of the cases written here is a stop; and a line that would be a row is
// one only if no call witnessed read a value out of a tensor (X).
pub(super) fn verdict(line: &[&str]) -> Taken {
    match (but_reads(line), line.get(X).and_then(|x| read_under(x))) {
        (Taken::Rows(_), Some(why)) => Taken::Stop(Box::leak(why.into_boxed_str())),
        (taken, _) => taken,
    }
}

// Where the column X is in a line.
const X: usize = 17;

// What a line means by every column but X.
pub(super) fn but_reads(line: &[&str]) -> Taken {
    let &[_, _, _, cpu, cuda, scpu, scuda, _, p, w, k, d, s, r, a, l, o, _, _] = line else { return Taken::Stop("its line of the table of torch's members has other columns than the rules read") };
    let is = |verdict: &str, word: &str| verdict.split('@').next() == Some(word);
    let stop = |why: String| Taken::Stop(Box::leak(why.into_boxed_str()));
    let enough = |n: &str| n.parse::<u32>().is_ok_and(|n| n >= 5);
    if s.starts_with("data") {
        stop(format!("the count or the sizes of its results follow the data of its operands, not only their keys (measured: {s})"))
    } else if o.contains("nested=data") {
        stop(format!("the operators that run under the call differ with the data of its operands, not only with their keys (measured: {o})"))
    } else if p.starts_with("yes") {
        Taken::Later("native")
    } else if !is(d, "no") {
        Taken::Later("ambient")
    } else if !is(w, "no") || !is(k, "no") {
        Taken::Later("writes")
    } else if r.starts_with("NO") {
        stop(format!("two calls on the same operands give different bits (measured: {r})"))
    } else if (l.contains("follows=") && !l.contains("follows=none")) || is(s, "no-sample") {
        Taken::Later("ambient")
    } else if a.starts_with("no-tensor-result") || l.contains("pyval=data") {
        Taken::Later("facts")
    } else if !is(s, "key") || !is(r, "yes") || !(is(p, "no") || p.starts_with("warn")) || !o.starts_with("nested=key") {
        stop(format!("its measured line is none the rules know (P {p}, S {s}, R {r}, O {o})"))
    } else if !enough(cpu) || !enough(cuda) {
        stop(format!("it has under 5 returning samples on a device (cpu {cpu}, cuda {cuda})"))
    } else if !enough(scpu) || !enough(scuda) {
        stop(format!("too few compared runs: the count and keys of its results were compared on other data in under 5 samples on a device (cpu {scpu}, cuda {scuda})"))
    } else if p.starts_with("warn") {
        Taken::Rows(&[ROW_WARN])
    } else {
        Taken::Rows(&[ROW])
    }
}

// Why a line whose column X is `x` is no row, or None. The sweep compares
// memory and results, and a value that is read and only decides whether the
// call raises changes neither: X is what the engine's witness of the
// dispatcher saw under one call per sample. Anything but `no` is no row:
// an operator handed a value out of a tensor (an error check on data, a
// number taken from a 0-dim tensor), or no call was witnessed, or the
// witness could not classify what ran.
pub(super) fn read_under(x: &str) -> Option<String> {
    match x.split('@').next() {
        Some("no") => None,
        Some(said) if said.starts_with("yes(") => Some(format!("its result, or whether it raises, depends on a value of a tensor: under calls of it an operator handed a value out of a tensor (measured: {x})")),
        _ => Some(format!("what runs under a call of it was not witnessed, or the witness could not classify it (measured: {x})")),
    }
}

// What the line of one form means for a member that only X stands against:
// its own line says of every call measured, whatever its form, that it only
// computes (`whole`: the row that line would give), and the lines of its
// forms say which form's calls read a value. A row if none of this form's
// did, on the samples a row needs on each device, EVERY one of them
// witnessed: X is `no@N`, N the calls that were witnessed, and a returning
// sample whose call was not witnessed (N under the samples of both devices)
// may be the one that reads. With all of them witnessed and 5 on each
// device, 5 calls were witnessed on each device.
pub(super) fn form_that_reads(line: &[&str], whole: &Row) -> Taken {
    let &[_, _, _, cpu, cuda, .., x, _] = line else { return Taken::Stop("its line of the table of torch's members has other columns than the rules read") };
    let stop = |why: String| Taken::Stop(Box::leak(why.into_boxed_str()));
    let enough = |n: &str| n.parse::<u32>().is_ok_and(|n| n >= 5);
    let samples = cpu.parse::<u32>().ok().zip(cuda.parse::<u32>().ok()).map(|(cpu, cuda)| cpu + cuda);
    let witnessed = x.split_once('@').and_then(|(_, n)| n.parse::<u32>().ok());
    if let Some(why) = read_under(x) {
        stop(why)
    } else if !enough(cpu) || !enough(cuda) {
        stop(format!("in this form it has under 5 returning samples on a device (cpu {cpu}, cuda {cuda})"))
    } else if samples.is_none() || witnessed != samples {
        stop(format!("in this form not every returning sample's call was witnessed (measured: {x}, of cpu {cpu} and cuda {cuda} samples): one that was not may read a value out of a tensor"))
    } else {
        Taken::Rows(Box::leak(Box::new([Row { reads: &[Read::Class], why: WHY_READS_FORM, before: Some(formed_call), ..*whole }])))
    }
}

// The operands a measured W names (`3(UNDECL;nover):22,4(...):22@60`), each
// with whether a schema declares the write; None: a W the rules do not read.
pub(super) fn written(w: &str) -> Option<&'static [Written]> {
    let said = w.rsplit_once('@')?.0;
    let mut found = vec![];
    for one in said.split(',').filter(|_| said != "no") {
        let (at, flags) = one.split_once('(')?;
        let declared = flags.split([';', ')']).next() == Some("decl");
        found.push(Written { at: Box::leak(at.to_string().into_boxed_str()), declared });
    }
    Some(Box::leak(found.into_boxed_slice()))
}

// Whether the changes of keys that a measured K names (`0(leaf
// requires_grad):2@22`) are what autograd makes of a declared in-place
// write: the written operand comes to require grad, as a result of the
// write and no longer a leaf, when another operand requires grad. Every
// in-place write does that (the samples of one member show it), and it is
// no change of the operand's sizes, strides or storage.
pub(super) fn of_the_write(k: &str, written: &[Written]) -> bool {
    let Some((said, _)) = k.rsplit_once('@') else { return false };
    said.split(',').all(|one| {
        let Some((at, fields)) = one.split_once('(') else { return false };
        let fields = fields.split(')').next().unwrap_or("");
        written.iter().any(|w| w.at == at && w.declared) && !fields.is_empty() && fields.split(' ').all(|f| f == "leaf" || f == "requires_grad")
    })
}

// What the line of a member that writes or re-keys an operand, or draws,
// means, the member's own (`form` "") or one form's: a row that names the
// operands written and says whether the calls drew (D: from the default
// generators; no form is of a call that hands a generator over), or a stop.
// The result is an operand where every call measured returned that operand
// itself (A).
pub(super) fn writer(line: &[&str], form: &'static str) -> Taken {
    let &[_, _, _, cpu, cuda, scpu, scuda, _, p, w, k, d, s, r, a, _, o, x, _] = line else { return Taken::Stop("its line of the table of torch's members has other columns than the rules read") };
    let is = |verdict: &str, word: &str| verdict.split('@').next() == Some(word);
    let stop = |why: String| Taken::Stop(Box::leak(why.into_boxed_str()));
    let enough = |n: &str| n.parse::<u32>().is_ok_and(|n| n >= 5);
    let warns = form.is_empty() && p.starts_with("warn");
    if o.contains("nested=data") {
        return stop(format!("the operators that run under the call differ with the data of its operands, not only with their keys (measured: {o})"));
    }
    let written = written(w);
    if !is(k, "no") && !written.is_some_and(|written| of_the_write(k, written)) {
        return stop(format!("it changes the key of an operand in place (measured: {k}), and what a region finds or checks by that operand's key would no longer hold"));
    }
    if !(is(p, "no") || warns) || !s.starts_with("key") || !r.starts_with("yes") || !o.starts_with("nested=key") {
        return stop(format!("its measured line is none the rules know for a member that writes or draws (P {p}, S {s}, R {r}, O {o})"));
    }
    if !enough(cpu) || !enough(cuda) {
        return stop(format!("it has under 5 returning samples on a device (cpu {cpu}, cuda {cuda})"));
    }
    if !enough(scpu) || !enough(scuda) {
        return stop(format!("too few compared runs: the count and keys of its results were compared on other data in under 5 samples on a device (cpu {scpu}, cuda {scuda})"));
    }
    if let Some(why) = read_under(x) {
        return stop(why);
    }
    let Some(written) = written else { return stop(format!("its measured line names what it writes in a way the rules do not read ({w})")) };
    let result = a.rsplit_once('@').and_then(|(a, _)| a.strip_prefix("r=is(")).filter(|a| !a.contains([';', '|', '/'])).and_then(|a| a.split_once(')')).and_then(|(k, _)| k.parse().ok()).map_or(Res::Entry, Res::Operand);
    let (writes, why) = if !is(d, "no") {
        (Write::Draws(written), WHY_DRAWS_FORM)
    } else {
        (if written.is_empty() { Write::None } else { Write::Operands(written) }, WHY_WRITES_FORM)
    };
    let row = Row { writes, result, enters: if warns { Enters::Foreign } else { Enters::Never }, form, ..ROW_WRITES };
    Taken::Rows(Box::leak(Box::new([if form.is_empty() { row } else { Row { reads: &[Read::Class], why, ..row } }])))
}
