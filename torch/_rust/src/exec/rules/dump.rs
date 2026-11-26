// The rule table as Python sees it: the dump, the walk of every kind, the
// groups of a class. A tool generates test programs from the dump: its
// format is regular, and is this.
//
// THE DUMP (_rules_dump): one line per entry, fields separated by tabs.
//   row    FAMILY  WHAT  OPERANDS  READS  WRITES  RESULT  ENTERS  SCALARS  WHY
//   stop   FAMILY  WHAT  WHY
//   later  FAMILY  WHAT  THE FAMILY THAT WILL TAKE IT
//   as     FAMILY  WHAT  KIND,KIND   (an opcode that is instructions of these kinds, or of none: -)
//   group  -       NAME [CLASS]  WHY
// OPERANDS is `-` or `POSITION:CLASS,...` (a position below 0 counts from
// the last operand); READS is `-` or names joined by commas.

use pyo3::ffi;
use pyo3::prelude::*;

use super::{groups, Taken, FAMILIES};

/// The rule table as text, one line per entry (the head of
/// ``exec/rules/dump.rs`` has the format).
#[pyfunction]
pub fn _rules_dump() -> String {
    let join = |parts: Vec<String>| if parts.is_empty() { "-".to_string() } else { parts.join(",") };
    let mut out = String::new();
    for family in FAMILIES {
        for (what, taken) in (family.table)() {
            let name = family.name;
            match taken {
                Taken::No | Taken::Unlisted => {}
                Taken::Later(other) => out += &format!("later\t{name}\t{what}\t{other}\n"),
                Taken::As(kinds) => out += &format!("as\t{name}\t{what}\t{}\n", join(kinds.iter().map(|k| format!("{k:?}")).collect())),
                Taken::Stop(why) => out += &format!("stop\t{name}\t{what}\t{why}\n"),
                Taken::Rows(rows) => {
                    for r in rows {
                        let operands = join(r.operands.iter().map(|(at, c)| format!("{at}:{}", unsafe { c.name() })).collect());
                        let reads = join(r.reads.iter().map(|x| format!("{x:?}")).collect());
                        out += &format!("row\t{name}\t{what}\t{operands}\t{reads}\t{:?}\t{:?}\t{:?}\t{:?}\t{}\n", r.writes, r.result, r.enters, r.scalars, r.why);
                    }
                }
            }
        }
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
    FAMILIES.iter().flat_map(|f| (f.table)()).map(|(what, taken)| (what, status(taken))).collect()
}

/// The groups of the rule table that the exact class of ``obj`` is in.
#[pyfunction]
pub fn _rules_groups(obj: &Bound<'_, PyAny>) -> Vec<String> {
    let tp = unsafe { ffi::Py_TYPE(obj.as_ptr()) };
    groups::GROUPS.iter().filter(|g| unsafe { g.holds(tp) }).map(|g| format!("{g:?}")).collect()
}
