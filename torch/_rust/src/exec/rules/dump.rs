// The rule table as Python sees it: the dump, the walk of every kind, the
// groups of a class. A tool generates test programs from the dump: its
// format is regular, and is this.
//
// THE DUMP (_rules_dump): everything license() decides, one line per entry,
// fields separated by tabs.
//   row    FAMILY  WHAT  OPERANDS  READS  WRITES  RESULT  ENTERS  SCALARS  WHY  BEFORE
//   stop   FAMILY  WHAT  WHY
//   later  FAMILY  WHAT  THE NAME IT IS LEFT UNDER
//   as     FAMILY  WHAT  KIND,KIND   (an opcode that is instructions of these kinds, or of none: -)
//   form   -       WHAT  FORM  row | stop | later:NAME  MEASURED   (a callee with lines of its own per form of
//                                 a call's operands: what a call of that form is answered, and what the form's
//                                 own line says, row | stop: they differ where the callee as a whole stops or
//                                 is left to a name, which is answered before any operand is looked at)
//   before -       NAME  FAMILY  WHAT IT REFUSES AND WHAT IT RECORDS   (a function rows name in BEFORE)
//   order  -       families | rows  VALUE  WHY   (in which order license() goes through the lines)
//   default callees | rules  CASE  later:NAME | stop  WHY   (what is answered where no line decides: a
//                                 callee no family lists, by its kind; and every other such case)
//   unheard -      NAME  WHY   (what a call taken at another's word may run with no rule heard)
//   group  -       NAME [CLASS]  WHY
// OPERANDS is `-` or `POSITION:CLASS,...` (a position below 0 counts from
// the last operand); READS is `-` or names joined by commas. A row that is
// for one form of a call's operands has WHAT as `CALLEE (FORM)`. WRITES of
// a call that writes operands is `Operands(AT,AT!,...)`: each the position
// of an operand among the call's (the receiver first) or the name of its
// keyword, with `!` where no schema declares the write. BEFORE is `-`, the
// NAME of a `before` line, or UNNAMED for a function no family's list has.
//
// THE DOOR (_rules_hear) asks rules::hear, the function the trace asks
// through, and answers for a row with that row's line of the dump, for a
// stop with the reason: a test asks about every line with operands made
// for it and holds the answer against the line. What the dump prints and
// what the door answers are formatted by one function from the row that
// license() chose, so the two agree only if license() chose that row.

use pyo3::ffi;
use pyo3::prelude::*;

use super::super::ffi::tr_current_frame;
use super::{callees, groups, Class, Heard, Kind, Row, Site, Taken, What, Write, BEFORES, DEFAULTS, FAMILIES, KINDS, ORDER, UNHEARD};

fn join(parts: Vec<String>) -> String {
    if parts.is_empty() { "-".to_string() } else { parts.join(",") }
}

// The line of the row `r` of `family` for `what`: as the dump prints it, and
// as the door answers with it.
fn row_line(family: &str, what: &str, r: &Row) -> String {
    let what = if r.form.is_empty() { what.to_string() } else { format!("{what} ({})", r.form) };
    let writes = match r.writes {
        Write::Operands(written) => format!("Operands({})", join(written.iter().map(|w| format!("{}{}", w.at, if w.declared { "" } else { "!" })).collect())),
        other => format!("{other:?}"),
    };
    let operands = join(r.operands.iter().map(|(at, c)| format!("{at}:{}", unsafe { c.name() })).collect());
    let reads = join(r.reads.iter().map(|x| format!("{x:?}")).collect());
    format!("row\t{family}\t{what}\t{operands}\t{reads}\t{writes}\t{:?}\t{:?}\t{:?}\t{}\t{}", r.result, r.enters, r.scalars, r.why, super::before_name(r))
}

// The line of the dump for the row license() chose for `asked`: the row is
// found among the rows the families have for it by what its line says.
fn line_of(asked: &What, row: &Row) -> String {
    let name = asked.name();
    for family in FAMILIES {
        if let Taken::Rows(rows) = (family.rule)(asked) {
            let line = row_line(family.name, &name, row);
            if rows.iter().any(|r| row_line(family.name, &name, r) == line) {
                return line;
            }
        }
    }
    "a row that no family has".to_string()
}

/// The rule table as text, one line per entry (the head of
/// ``exec/rules/dump.rs`` has the format).
#[pyfunction]
pub fn _rules_dump() -> String {
    let mut out = String::new();
    for family in FAMILIES {
        for (what, taken) in (family.table)() {
            let name = family.name;
            match taken {
                Taken::No | Taken::Unlisted => {}
                Taken::Later(other) => out += &format!("later\t{name}\t{what}\t{other}\n"),
                Taken::As(kinds) => out += &format!("as\t{name}\t{what}\t{}\n", join(kinds.iter().map(|k| format!("{k:?}")).collect())),
                Taken::Stop(why) => out += &format!("stop\t{name}\t{what}\t{why}\n"),
                Taken::Rows(rows) => rows.iter().for_each(|r| out += &format!("{}\n", row_line(name, &what, r))),
            }
        }
    }
    // A form's own line is heard only where the callee has rows: a callee
    // that stops, or is left to a name, is answered before any operand is
    // looked at.
    let torch = super::torch::FAMILY;
    let members: std::collections::HashMap<String, Taken> = (torch.table)().into_iter().collect();
    for (member, form, measured) in super::torch::_rules_torch_forms() {
        let answer = match members.get(&member) {
            Some(Taken::Rows(_)) => measured.clone(),
            Some(Taken::Later(name)) => format!("later:{name}"),
            _ => "stop".to_string(),
        };
        out += &format!("form\t-\t{member}\t{form}\t{answer}\t{measured}\n");
    }
    for (family, list) in BEFORES {
        for (_, name, says) in *list {
            out += &format!("before\t-\t{name}\t{family}\t{says}\n");
        }
    }
    out += &format!("order\t-\tfamilies\t{}\t{}\n", join(FAMILIES.iter().map(|f| f.name.to_string()).collect()), ORDER[0]);
    out += &format!("order\t-\trows\tfirst\t{}\n", ORDER[1]);
    for (kind, taken, why) in callees::KINDS {
        match taken {
            Taken::Later(family) => out += &format!("default\tcallees\t{kind}\tlater:{family}\t{why}\n"),
            Taken::Stop(_) => out += &format!("default\tcallees\t{kind}\tstop\t{why}\n"),
            _ => {}
        }
    }
    for (case, answer, why) in DEFAULTS {
        out += &format!("default\trules\t{case}\t{answer}\t{why}\n");
    }
    for (family, why) in UNHEARD {
        out += &format!("unheard\t-\t{family}\t{why}\n");
    }
    for (name, why) in groups::GROUPS.iter().flat_map(|g| unsafe { g.sentences() }) {
        out += &format!("group\t-\t{name}\t{why}\n");
    }
    out
}

/// Every instruction kind and every opcode number with what the rule table
/// answers for it: ``(name, "row" | "stop" | "later:FAMILY" | "as:KIND,..."
/// | "unlisted")``.
#[pyfunction]
pub fn _rules_kinds() -> Vec<(String, String)> {
    let status = |taken: Taken| match taken {
        Taken::No | Taken::Unlisted => "unlisted".to_string(),
        Taken::Later(family) => format!("later:{family}"),
        Taken::As(kinds) => format!("as:{}", kinds.iter().map(|k| format!("{k:?}")).collect::<Vec<_>>().join(",")),
        Taken::Stop(_) => "stop".to_string(),
        Taken::Rows(_) => "row".to_string(),
    };
    // (The torch family lists callees, not kinds: _rules_torch has them.)
    FAMILIES.iter().filter(|f| f.name != super::torch::FAMILY.name).flat_map(|f| (f.table)()).map(|(what, taken)| (what, status(taken))).collect()
}

/// The groups of the rule table that the exact class of ``obj`` is in.
#[pyfunction]
pub fn _rules_groups(obj: &Bound<'_, PyAny>) -> Vec<String> {
    let tp = unsafe { ffi::Py_TYPE(obj.as_ptr()) };
    groups::GROUPS.iter().filter(|g| unsafe { g.holds(tp) }).map(|g| format!("{g:?}")).collect()
}

/// What a call taken at another's word may run without a rule being heard:
/// ``(family, why)`` for each family of rules that is not written yet and is
/// on that list.
#[pyfunction]
pub fn _rules_unheard() -> Vec<(String, String)> {
    UNHEARD.iter().map(|(family, why)| (family.to_string(), why.to_string())).collect()
}

/// What the trace of today's engine hears from the rule table about an
/// instruction before it runs it. What the instruction is, is said by
/// exactly one of ``kind`` (the name of an instruction kind), ``opcode``
/// (the number of an opcode, with ``oparg``) and ``callee`` (the callee of a
/// call: any object, taken as it is); ``operands`` are the instruction's, in
/// order, and ``kwnames`` the names of a call's keywords (the last operands);
/// ``at_a_word``: it runs under a call taken at another's word; ``names``:
/// the names of the code an oparg counts into, with ``frame`` true the frame
/// of the caller is where it runs (a row that reads the frame is asked with
/// no frame otherwise, and says no); ``make`` false: the tables a callee is
/// told by are not made first. ``("stop", why)``, ``("row", the row's line
/// of the dump)``, ``("older", name)`` (left to the code that decided before
/// the table, and by name to the rules that will take it) or ``("unheard",
/// name)``. For tests; the object ``None`` cannot be asked about as a callee
/// here (it means "no callee").
#[pyfunction]
#[pyo3(signature = (*, kind=None, opcode=None, callee=None, operands=Vec::new(), kwnames=None, at_a_word=false, oparg=0, names=Vec::new(), frame=false, make=true))]
#[allow(clippy::too_many_arguments)]
pub fn _rules_hear(
    kind: Option<String>,
    opcode: Option<u8>,
    callee: Option<&Bound<'_, PyAny>>,
    operands: Vec<Py<PyAny>>,
    kwnames: Option<&Bound<'_, pyo3::types::PyTuple>>,
    at_a_word: bool,
    oparg: u32,
    names: Vec<Py<PyAny>>,
    frame: bool,
    make: bool,
) -> PyResult<(String, String)> {
    let asked = match (kind, opcode, callee) {
        (Some(name), None, None) => {
            let kind: Option<&Kind> = KINDS.iter().find(|k| format!("{k:?}") == name);
            What::Op(*kind.ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("{name} is no instruction kind")))?)
        }
        (None, Some(opcode), None) => What::Opcode(opcode, oparg),
        (None, None, Some(callee)) => What::Callee(callee.as_ptr()),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("exactly one of kind, opcode and callee says what is asked about")),
    };
    if make {
        // (The tables a callee is told by are made where a function goes
        // onto the executor; a test may ask before any did.)
        super::ready();
    }
    let operands: Vec<*mut ffi::PyObject> = operands.iter().map(|o| o.as_ptr()).collect();
    let names: Vec<*mut ffi::PyObject> = names.iter().map(|o| o.as_ptr()).collect();
    let classes: Vec<Class> = operands.iter().map(|o| unsafe { Class::of(*o) }).collect();
    let site = Site {
        frame: if frame { unsafe { tr_current_frame(ffi::PyThreadState_Get()) } } else { std::ptr::null_mut() },
        names: if names.is_empty() { std::ptr::null() } else { names.as_ptr() },
        operands: &operands,
        kwnames: kwnames.map_or(std::ptr::null_mut(), |k| k.as_ptr()),
    };
    Ok(match unsafe { super::hear(&asked, &classes, &site, at_a_word) } {
        Heard::Stop(why) => ("stop".into(), why),
        Heard::Row(row) => ("row".into(), line_of(&asked, row)),
        Heard::Older => ("older".into(), super::later(&asked).unwrap_or("").into()),
        Heard::Unheard(family) => ("unheard".into(), family.into()),
    })
}

/// The kind of callable the rule table takes ``callee`` for where no family
/// has a line for it (a kind of ``exec/rules/callees.rs``).
#[pyfunction]
pub fn _rules_callee_kind(callee: &Bound<'_, PyAny>) -> String {
    super::ready();
    unsafe { callees::kind_name(callee.as_ptr()) }.to_string()
}
