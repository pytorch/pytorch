// The measured domain of a member with a row (see the head of
// rules/torch.rs): the classes its returning samples had, place by place,
// and the switches that enforce or report it.

use super::*;

// The column `domain` of a member's line: `place=class|class ...`.
pub(super) fn domain(line: &[&'static str]) -> Vec<(&'static str, Vec<&'static str>)> {
    line.last().into_iter().flat_map(|places| places.split(' ')).filter_map(|place| place.split_once('=')).map(|(place, classes)| (place, classes.split('|').collect())).collect()
}

// The two switches of the measured domain (the head of the file), read once
// in ready(): whether a call outside it has no row, and the file such calls
// are written to.
pub(super) struct Domain {
    enforced: bool,
    report: Option<String>,
}

pub(super) static DOMAIN: OnceLock<Domain> = OnceLock::new();

pub(super) fn domain_switches() -> &'static Domain {
    DOMAIN.get_or_init(|| Domain { enforced: std::env::var("TORCH_RUST_RULES_DOMAIN").is_ok_and(|v| v == "1"), report: std::env::var("TORCH_RUST_RULES_UNSAMPLED").ok().filter(|f| !f.is_empty()) })
}

// Whether every operand of a call of `member` is of a class that a
// returning sample of the member had in that place. Err: the first place
// where not, and the class. Asked only with a switch of the domain set.
pub(super) unsafe fn sampled(member: &Member, operands: &[*mut PyObject], kwnames: *mut PyObject) -> Result<(), String> {
    let Some(switches) = DOMAIN.get().filter(|d| d.enforced || d.report.is_some()) else { return Ok(()) };
    for (place, o) in placed(operands, kwnames)? {
        let class = form(o, 0);
        let had = member.domain.iter().find(|(named, _)| *named == place).map(|(_, classes)| classes);
        if !class.as_deref().is_some_and(|class| had.is_some_and(|had| had.contains(&class))) {
            let class = class.unwrap_or_else(|| reads::type_name(o).to_string());
            if let Some(file) = &switches.report {
                use std::io::Write as _;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(file) {
                    let _ = writeln!(f, "{}\t{place}\t{class}", member.name);
                }
            }
            if switches.enforced {
                return Err(format!("operand {place} is a {class}, and no call of it that was measured had an operand of that class there"));
            }
        }
    }
    Ok(())
}
