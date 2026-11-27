// The torch family: torch's own C callees, by what calls of them were
// measured to do.
//
// torch_members.tsv (generated: torch/_rust/tools/gen_torch_members.py) has
// one line per C callee with the verdicts of a sweep that called it on many
// operands, on the CPU and on CUDA, and looked at memory: whether a Python
// frame was entered under the call (P), an operand's bytes or key changed
// (W, K), a generator moved (D), the results' count and keys were the same
// for other data in operands of the same keys (S), two calls gave the same
// bits (R). No name and no schema is asked. What a line means is verdict()
// below, and nothing else:
//   - a ROW for a member that only computes: P no (or only to show a
//     warning: those frames are no part of the row and come to the gate as
//     any frame), W, K, D no, S key, R yes, at least 5 returning samples on
//     each device. The row is for a call whose operands are plain (before):
//     exact Tensor / Parameter, the immutable classes, torch's constants,
//     exact tuples, lists and dicts of those, and no `out=`;
//   - a STOP, with the measured verdict, for a member whose results follow
//     data, under which the operators run differ with the data (O), whose
//     two calls differ, or that was not measured enough: under 5 returning
//     samples on a device, or under 5 whose results were compared on other
//     data (what S rests on). No trace runs it, wherever it is called;
//   - THE MEASURED DOMAIN (built, not enforced). The column `domain` has
//     the classes of operands the member's returning samples had, place by
//     place (a position, the receiver first, or a keyword's name). A call
//     that hands an operand of another class in some place was not
//     measured. With TORCH_RUST_RULES_DOMAIN=1 in the environment such a
//     call has no row; with TORCH_RUST_RULES_UNSAMPLED=FILE it is written to
//     FILE as `member, place, class`. Without them nothing is asked: the
//     calls that real programs make are not all sampled yet;
//   - a ROW PER FORM for a member that only its results following data
//     stands against, where the samples tell its forms apart. A FORM is
//     which places among a call's operands hold a tensor, or an exact list
//     or tuple with a tensor in it (`0:T 1:T`, `0:[T]`; `plain`: none): a
//     number is read once and is the same on every run of a region, the
//     data of a tensor is not (a 0-dim tensor where an int is declared is
//     read by the argument parser, and the sizes of the result follow it).
//     Behind such a member's line the file has a line per form, computed
//     from the samples of that form alone and read by the same verdict(). A
//     call of a form whose line is a row has the row; a call of any other
//     form, measured or not, stops the trace (formed_call). A member none of
//     whose forms is a row, and whose calls that hand no tensor (`plain`: a
//     factory's) are left to another family, is left to that family whole;
//   - for a member that WRITES the memory of an operand and draws nothing
//     (writer): a row that names the operands written, for any plain call,
//     if every write measured is one the schema of its operator declares and
//     no call measured changed an operand's key; else a row per form, each
//     on its form's own samples, with a bool operand in the form by its
//     value where the calls measured do not do the same under both
//     (batch_norm writes its running statistics only in training, and no
//     schema says so). A form in which an operand's KEY was changed in place
//     (resize_, set_, requires_grad_, squeeze_, unsqueeze_, t_) is a stop:
//     what a region finds or checks by that operand's key would no longer
//     hold. Not so the one change every in-place write makes (of_the_write:
//     the written operand comes to require grad when another operand does;
//     copy_ has the samples): the code that was there before the table
//     knows in-place writes under autograd, and a row takes none of it
//     away. The row SAYS what is written
//     (Write::Operands) and does nothing about it: whoever runs the call
//     asks (the trace of today's engine through hand_over, from
//     covered::step, before the call runs: the trace's one record of its
//     writes is told of each tensor, torch/_rust_inductor/undo.py; given
//     up, the trace undoes its writes, the last first, and it no longer
//     takes the call for one that leaves nothing behind);
//   - for a member that DRAWS from a random generator and that nothing
//     else stands against (D; S key, R yes: two calls from the same
//     generator state give the same bits): a row per form, as for a writer
//     with forms, each on its form's own samples. The row of a form whose
//     calls drew says so (Write::Draws, with the operands it writes
//     besides: normal_ writes its receiver); a form whose calls drew
//     nothing is a row like any other (dropout with train=False returns its
//     operand). The generators are the default ones: a call that hands a
//     generator over has no row (plain_call: a generator is no plain
//     operand; and no form is of such a call). Whoever runs the call asks
//     (hand_over: today's trace records the generators' states and puts
//     them back if it is given up; only_computes is false for the member);
//   - NO ROW for a line under whose calls a VALUE was read out of a tensor
//     (X: what the engine's witness of the dispatcher saw under one call per
//     sample; anything but `no` is no row, so also a line no call of which
//     was witnessed). The sweep compares memory and results, and a value
//     that only decides whether the call raises (the error check of a
//     decomposition), or that the argument parser takes from a 0-dim tensor
//     where a number is declared, changes neither. A member that only this
//     stands against has a line per form too, and goes by its forms: its
//     own line says of every call measured that it only computes, the line
//     of a form says whether calls of that form read (form_that_reads). A
//     form in which they do, or with too few samples, or with no line,
//     stops the trace; a member that reads in every form stops whole;
//   - A LINE PER SET OF KEYWORDS for a form that only such a read stands
//     against, where the keywords a call passes tell the calls that read
//     from those that do not (the parser takes another overload of the
//     operator: _foreach_add_(list, t) reads the 0-dim tensor's value as a
//     number, _foreach_add_(list, t, alpha=1.0) hands the tensor on). The
//     lines then name the keywords that hold no tensor (`0:[T] 1:T
//     alpha:K`; `-:K`: none), each is on the samples of its set alone, and
//     a call fits one only if it passes exactly those keywords (fits): a
//     call with a set of keywords that was not measured apart stops the
//     trace. A form whose lines tell no keyword is of the calls with any;
//   - LATER, with the family that will take it, for a member that draws
//     and has no line of a form, makes a tensor of no tensor or
//     follows ambient state (ambient), returns a fact or a value of a tensor
//     (facts), or enters Python code under the call (native: on CUDA seven
//     members go through the Python router of torch._native, whose frames
//     covered.rs still takes for part of the operator by their path): the
//     code that decided before decides.
// A torch callee that is not in the file is not this family's (Taken::No):
// the name filter of shown.rs still decides for it.
//
// Where what is: this file reads the table and resolves its members;
// torch/lines.rs is what a measured line means (verdict, writer);
// torch/forms.rs the operands of a call (plain_call, formed_call);
// torch/sampled.rs the measured domain; torch/writes.rs what a call with a
// row writes, and the hand-over of it to today's trace.
//
// What it is given
//   - by rules::rows, a callee as the instruction has it: the callable, a
//     builtin bound to its receiver, a functools.partial of one;
//   - by ready(), called where a function goes onto the executor and never
//     under a trace: the moment to resolve each member's name to its object
//     (attribute loads on torch's own namespaces, in Python, once).
// What it guarantees
//   - a callee is matched by identity: a builtin and a method descriptor by
//     the C definition they are made from (so `x.add` bound to a tensor is
//     `Tensor.add`), anything else by its address;
//   - under a trace rule() and the rows' `before` run no Python code: they
//     read the table, the constants of torch and the switches made by
//     ready(), never make them (members() is called by ready(), by the dump
//     and by the doors for tests only);
//   - with no table (ready() did not run, or failed, or a name of the table
//     resolves to nothing in this torch) every callee is a stop.
// What it may not assume
//   - that a row is all the license a call has: the checks that were there
//     before the table still run behind it.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::super::ffi::{tr_bound_function, tr_partial_func};
use super::super::{reads, scalars};
use super::{Before, Enters, Family, Group, Read, Res, Row, Scalars, Site, Taken, What, Write, Written};

mod forms;
mod lines;
mod sampled;
mod writes;

use forms::*;
use lines::*;
use sampled::*;
pub use writes::{hand_over, only_computes};
use writes::*;

pub const FAMILY: Family = Family { name: "torch", rule, table };

const WHY: &str = "Measured on the CPU and on CUDA: the call enters no Python frame, changes no operand's bytes or key, moves no generator, gives the same bits twice, and the count and keys of its results are the same for other data in operands of the same keys. It is run again on each call's operands.";
const WHY_WARN: &str = "As a member that only computes, but it can show a warning: the frames of that are not the row's, and are asked about as any frame under the body.";
const ROW: Row = Row { operands: &[], reads: &[], writes: Write::None, result: Res::Entry, enters: Enters::Never, scalars: Scalars::Operand, why: WHY, before: Some(plain_call), form: "" };
const WHY_FORM: &str = "As a member that only computes, for the forms of operands measured so, each on its own samples: in another form of operands the count or the sizes of its results follow data, or it was not measured, and a call of that form stops the trace.";
const ROW_FORM: Row = Row { operands: &[], reads: &[Read::Class], writes: Write::None, result: Res::Entry, enters: Enters::Never, scalars: Scalars::Operand, why: WHY_FORM, before: Some(formed_call), form: "" };
const WHY_READS_FORM: &str = "As a member that only computes, for the forms of operands in which no call witnessed read a value out of a tensor: in another form its result, or whether it raises, follows a value of a tensor (a 0-dim tensor where a number is declared, an error check on data), and a call of that form stops the trace.";
const ROW_WARN: Row = Row { operands: &[], reads: &[], writes: Write::None, result: Res::Entry, enters: Enters::Foreign, scalars: Scalars::Operand, why: WHY_WARN, before: Some(plain_call), form: "" };
const WHY_WRITES: &str = "Measured on the CPU and on CUDA: the call writes the memory of the operands named and no other, changes no operand's key, moves no generator and gives the same bits twice; every write measured is one the schema of its operator declares, on the same operands in every form of the call that was measured.";
const WHY_WRITES_FORM: &str = "As a member that writes operands, for one form of the call's operands (a bool among them by its value) on that form's own samples: which operands the call writes, declared by a schema or not, is what was measured in this form. A call of a form with no row stops the trace.";
const WHY_DRAWS_FORM: &str = "As a member that writes operands, for one form of the call's operands (a bool among them by its value) on that form's own samples: the calls of this form move the state of a default random generator (of the CPU, or of a CUDA device), and write the operands named. No call of the form hands a generator over. Run again, the call draws again: a list with it is not run a second time on one arrival, and a trace that is given up puts the generators back.";
const ROW_WRITES: Row = Row { operands: &[], reads: &[], writes: Write::None, result: Res::Entry, enters: Enters::Never, scalars: Scalars::Operand, why: WHY_WRITES, before: Some(writing_call), form: "" };

// The `before` functions of this family's rows, each by the function it
// is, with what it refuses and what it records: the dump names a row's
// `before` by this list, and a `before` that is not on it has no name.
pub const BEFORES: &[(Before, &str, &str)] = &[
    (plain_call, "plain_call", "refuses: a callee that is no member of the table; a call through a functools.partial or through a slot bound to its receiver; a builtin bound to a class or module that is not exactly a type or a module; `out=`; an operand that is not plain (an exact Tensor or Parameter, an immutable of the closed list, a constant of torch, the engine's stand-in for a number, an exact tuple, list or dict with str keys of those, 8 deep at most); with the domain switch on, an operand of a class no sample of the member had in its place. records: nothing (the operands are what the call is run again on)"),
    (formed_call, "formed_call", "refuses: what plain_call refuses; a dict among the operands, or containers nested deeper than a form tells; a call whose form (the places that hold a tensor or an exact list or tuple with one, each bool the form names by its value, and, where the lines of the form tell keywords, exactly the keywords it passes) has no line that is a row, or has a row of its own that is not this one. records: nothing"),
    (writing_call, "writing_call", "refuses: what plain_call refuses, or formed_call if the row is for one form; a call that does not hand over, in its place, an operand the row says is written. records: nothing, and tells nobody: the tensors the call writes, and that it draws from the default generators, go to the trace's record of writes and draws through hand_over, which the executor calls behind the license"),
];

// Resolves `names` as attributes from a loaded module on: `found[i]` is the
// object, or None. Of a class it is the C member that the class or a base
// has itself (a Python method of the same name may hide it: Tensor.unflatten).
const RESOLVE: &CStr = c"
import sys
C = ('builtin_function_or_method', 'method_descriptor', 'wrapper_descriptor', 'classmethod_descriptor', 'getset_descriptor')
def at(name):
    parts = name.removesuffix('.__get__').split('.')
    found = sys.modules[parts[0]]
    for part in parts[1:]:
        own = [vars(c)[part] for c in found.__mro__ if part in vars(c)] if isinstance(found, type) else []
        own = [member for member in own if type(member).__name__ in C]
        found = own[0] if own else getattr(found, part)
    return found
found = []
for name in names:
    try:
        found.append(at(name))
    except Exception:
        found.append(None)
";

struct Member {
    name: &'static str,
    taken: Taken,
    // The object the name resolved to, by address (0: it did not).
    object: usize,
    // The lines of its forms, if it goes by them: a row or a stop each.
    forms: Vec<(&'static str, Taken)>,
    // Whether what a call of it is goes by the form of the call's operands.
    by_form: bool,
    // Whether its line says that it writes or re-keys an operand.
    writer: bool,
    // The row its own line would give if no call of it read a value out of
    // a tensor, where only that stands against it: its forms say which do.
    but_reads: Option<&'static Row>,
    // The classes its returning samples had in each place among their
    // operands (a position, or a keyword's name).
    domain: Vec<(&'static str, Vec<&'static str>)>,
}

struct Members {
    all: Vec<Member>,
    // What identifies a member's C function (see key) -> its index.
    by_key: HashMap<usize, usize>,
    failed: bool,
}

static MEMBERS: OnceLock<Members> = OnceLock::new();

// What identifies the C function behind a callable: for a builtin and a
// method descriptor the definition both are made from, else the object.
unsafe fn key(f: *mut PyObject) -> usize {
    let tp = ffi::Py_TYPE(f);
    if tp == &raw mut ffi::PyCFunction_Type {
        (*(f as *mut ffi::PyCFunctionObject)).m_ml as usize
    } else if tp == &raw mut ffi::PyMethodDescr_Type || tp == &raw mut ffi::PyClassMethodDescr_Type {
        (*(f as *mut ffi::PyMethodDescrObject)).d_method as usize
    } else {
        f as usize
    }
}

// The table, made on first use: runs Python code (RESOLVE).
fn members() -> &'static Members {
    MEMBERS.get_or_init(|| {
        let lines: Vec<Vec<&'static str>> = include_str!("torch_members.tsv").lines().filter(|l| !l.starts_with('#') && !l.is_empty()).skip(1).map(|l| l.split('\t').collect()).collect();
        // A form's line comes behind its member's: a row, or a stop (what a
        // member's line leaves to another family is no row of a form).
        let mut all: Vec<Member> = vec![];
        for line in &lines {
            match (line.get(1), all.last_mut()) {
                // A member that writes: by its own line if that says of
                // every write that a schema declares it and of no key that
                // it changed (the lines of its forms then say no more);
                // else by the lines of its forms, which follow.
                (Some(&"-"), _) => all.push(match verdict(line) {
                    Taken::Later("writes") => {
                        let whole = writer(line, "");
                        let declared = matches!(whole, Taken::Rows(rows) if matches!(rows[0].writes, Write::Operands(written) if written.iter().all(|w| w.declared)));
                        Member { name: line[0], taken: whole, object: 0, forms: vec![], by_form: !declared, writer: true, but_reads: None, domain: domain(line) }
                    }
                    // A member that draws, and nothing else stands against
                    // it: by the lines of its forms, read as a writer's (in
                    // which form it draws, and what it writes besides). With
                    // no line of a form it is left as it was.
                    Taken::Later("ambient") if line.get(11).is_some_and(|d| d.split('@').next() != Some("no")) => Member { name: line[0], taken: Taken::Later("ambient"), object: 0, forms: vec![], by_form: true, writer: true, but_reads: None, domain: domain(line) },
                    taken => {
                        let but_reads = match (taken, but_reads(line)) {
                            (Taken::Stop(_), Taken::Rows(rows)) => rows.first(),
                            _ => None,
                        };
                        Member { name: line[0], taken, object: 0, forms: vec![], by_form: matches!(taken, Taken::Stop(_)), writer: false, but_reads, domain: domain(line) }
                    }
                }),
                (Some(_), Some(member)) if member.name == line[0] && !member.by_form => {}
                (Some(form), Some(member)) if member.name == line[0] && member.writer => member.forms.push((form, writer(line, form))),
                (Some(form), Some(Member { name, forms, but_reads: Some(whole), .. })) if *name == line[0] => forms.push((form, form_that_reads(line, whole))),
                (Some(form), Some(member)) if member.name == line[0] => member.forms.push((form, match verdict(line) {
                    Taken::Rows(_) => Taken::Rows(&[ROW_FORM]),
                    Taken::Stop(why) => Taken::Stop(why),
                    // Calls that hand no tensor (a factory's): nothing to
                    // compare, and another family's to take.
                    Taken::Later(family) if *form == "plain" => Taken::Later(family),
                    _ => Taken::Stop("in this form it was not measured to only compute"),
                })),
                _ => all.push(Member { name: line[0], taken: Taken::Stop("its line of the table of torch's members is of no member"), object: 0, forms: vec![], by_form: false, writer: false, but_reads: None, domain: vec![] }),
            }
        }
        // A member that goes by its forms has the rows of its forms. One
        // that has none, and whose calls that hand no tensor are left to
        // another family (a factory), is left to that family whole, as
        // before its forms were measured apart: the code that decided
        // before decides until that family takes it.
        for member in all.iter_mut().filter(|m| m.by_form && !m.forms.is_empty()) {
            let rows: Vec<Row> = member.forms.iter().filter_map(|(_, taken)| if let Taken::Rows(rows) = taken { Some(Row { ..rows[0] }) } else { None }).collect();
            let later = member.forms.iter().find_map(|(_, taken)| if let Taken::Later(family) = taken { Some(*family) } else { None });
            if !rows.is_empty() {
                // A row per form of a member that writes; one row for all
                // the forms of one that only computes (theirs are the same).
                member.taken = Taken::Rows(Box::leak(if member.writer { rows } else { rows.into_iter().take(1).collect() }.into_boxed_slice()));
            } else if let (Some(family), false) = (later, member.writer) {
                member.taken = Taken::Later(family);
            } else if member.writer && !matches!(member.taken, Taken::Stop(_)) {
                member.taken = Taken::Stop("no form of its operands that was measured apart has a row");
            }
        }
        let mut by_key = HashMap::new();
        let failed = Python::attach(|py| unsafe {
            let g = pyo3::types::PyDict::new(py);
            let ran = g.set_item("names", all.iter().map(|member| member.name).collect::<Vec<_>>()).and_then(|_| py.run(RESOLVE, Some(&g), None));
            let found = reads::dict_get(g.as_ptr(), ffi::PyUnicode_InternFromString(c"found".as_ptr()));
            if ran.is_err() || found.is_null() || ffi::PyList_GET_SIZE(found) != all.len() as ffi::Py_ssize_t {
                return true;
            }
            // A name that resolves to nothing here: the table is not of
            // this torch, and no callee can be told to be none of its
            // members (one of its stops would be left to other code).
            let mut unresolved = false;
            for (i, member) in all.iter_mut().enumerate() {
                let object = ffi::PyList_GET_ITEM(found, i as ffi::Py_ssize_t);
                if object != ffi::Py_None() {
                    member.object = object as usize;
                    by_key.insert(key(object), i);
                } else {
                    unresolved = true;
                }
            }
            if unresolved {
                return true;
            }
            // The table refers to its objects by address: kept for good.
            std::mem::forget(g.unbind());
            false
        });
        Members { all, by_key, failed }
    })
}

// Makes the table. Runs Python code: called where a function goes onto the
// executor, not from inside it.
pub fn ready() {
    members();
    super::super::torch_constant_types();
    domain_switches();
}

// The member `f` calls: itself, the method a bound builtin is of, or what a
// functools.partial calls.
unsafe fn member(f: *mut PyObject) -> Option<&'static Member> {
    // (Read, never made here: members() makes it, from ready().)
    let made = MEMBERS.get()?;
    let partial = tr_partial_func(f);
    let called = tr_bound_function(if partial.is_null() { f } else { partial });
    made.by_key.get(&key(called)).map(|i| &made.all[*i])
}

// The name of the member `f` calls, for a refusal.
pub unsafe fn name(f: *mut PyObject) -> Option<&'static str> {
    member(f).map(|m| m.name)
}

fn rule(what: &What) -> Taken {
    let What::Callee(f) = *what else { return Taken::No };
    match MEMBERS.get() {
        None => Taken::Stop("the table of torch's members is not made"),
        Some(made) if made.failed => Taken::Stop("the table of torch's members could not be made"),
        Some(_) => unsafe { member(f) }.map_or(Taken::No, |m| m.taken),
    }
}

fn table() -> Vec<(String, Taken)> {
    members().all.iter().map(|m| (m.name.to_string(), m.taken)).collect()
}

/// The `before` functions of the family's rows: for each of BEFORES its
/// name, what it refuses and records, and how many rows of the table have
/// it as their `before`; last, under the name ``"UNNAMED"``, how many rows
/// have a `before` that is on no list.
#[pyfunction]
pub fn _rules_torch_befores() -> Vec<(String, String, usize)> {
    let rows = members().all.iter().flat_map(|m| if let Taken::Rows(rows) = m.taken { rows.iter() } else { [].iter() });
    let befores: Vec<usize> = rows.filter_map(|row| row.before.map(|f| f as usize)).collect();
    let count = |f: Before| befores.iter().filter(|b| **b == f as usize).count();
    let mut found: Vec<(String, String, usize)> = BEFORES.iter().map(|(f, name, says)| (name.to_string(), says.to_string(), count(*f))).collect();
    let named: usize = found.iter().map(|(_, _, rows)| rows).sum();
    found.push(("UNNAMED".into(), String::new(), befores.len() - named));
    found
}

/// Every member of the table of torch's C callees: its name, the address of
/// the object the name resolved to (0: it did not), and what the rules say
/// of it: ``"row"``, ``"stop"`` or ``"later:FAMILY"``.
#[pyfunction]
pub fn _rules_torch() -> Vec<(String, usize, String)> {
    members().all.iter().map(|m| (m.name.to_string(), m.object, status(m.taken))).collect()
}

fn status(taken: Taken) -> String {
    match taken {
        Taken::Rows(_) => "row".to_string(),
        Taken::Later(family) => format!("later:{family}"),
        _ => "stop".to_string(),
    }
}

/// Every form of operands the table has a line of its own for: the member,
/// the form, and what the rules say of a call of that form: ``"row"`` or
/// ``"stop"``.
#[pyfunction]
pub fn _rules_torch_forms() -> Vec<(String, String, String)> {
    members().all.iter().flat_map(|m| m.forms.iter().map(|(form, taken)| (m.name.to_string(), form.to_string(), status(*taken)))).collect()
}

/// The classes of operands each member of the table was measured on:
/// ``(member, place, classes)`` for every place among the operands (a
/// position, the receiver first, or the name of a keyword) that a returning
/// sample of the member had an operand in.
#[pyfunction]
pub fn _rules_torch_domain() -> Vec<(String, String, Vec<String>)> {
    members().all.iter().flat_map(|m| m.domain.iter().map(|(place, classes)| (m.name.to_string(), place.to_string(), classes.iter().map(|c| c.to_string()).collect()))).collect()
}
