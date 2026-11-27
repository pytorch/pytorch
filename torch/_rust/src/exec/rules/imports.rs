// The imports: IMPORT_NAME and IMPORT_FROM, the first rows that record what
// they read themselves.
//
// `import m` in a function asks the frame's builtins for `__import__` and
// calls it. The interpreter's own, for a module that is loaded, returns an
// item of the table of loaded modules (sys.modules) after looking at whether
// the module is still being initialized, and does nothing else: no code of
// the program runs and nothing is written. That case has a row. What the row
// read is recorded as checks of the region being traced, so that a region
// holding the import is entered only while each of those reads gives what it
// gave: another object under the name in sys.modules, the name gone, or
// another `__import__` in the builtins is a miss.
//
// Everything else an import can do has no row and stops the trace: a module
// that is not loaded or is being initialized (the import machinery, the
// module's own code), an `__import__` that is not the interpreter's, a
// relative import (the package is found through the frame's globals), names
// taken from a package (importlib's _handle_fromlist, Python code), a module
// with a `__getattr__` of its own where the import would ask it, an object in
// sys.modules that is no exact module.
//
// What it is given
//   - by rules::license, once the classes of the operands fit a row: the
//     opcode with its oparg (the index of the name) and the Site.
// What it guarantees
//   - before() is Ok only if the instruction, run next on these operands,
//     takes the path described above, and then every object that path reads
//     is an entry of the trace (exec/reads.rs): the `__import__` item of the
//     builtins dict, the items of sys.modules, `__spec__` of each module
//     found and `_initializing` of that spec (False in the spec's own dict,
//     or in no place the lookup has: see loaded()), the items of the module's
//     dict the import asks for, each present with its value or absent;
//   - it runs no code of the program: classes are compared by address, dicts
//     with str keys are read in C, a new str is made for a dotted name.
// What it may not assume
//   - that the level, the list of names or the state of a module can be read
//     off a class: they are VALUES, and are tested here, in before(), not in
//     the row's key.

use std::ffi::CStr;
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;

use super::super::ffi::{tr_attr_home, tr_frame_builtins, tr_type_lookup};
use super::super::reads;
use super::{Class, Enters, Family, Group, Read, Res, Row, Scalars, Site, Taken, What, Write};
use crate::cfg::opcodes::{self, IMPORT_FROM, IMPORT_NAME};

pub const FAMILY: Family = Family { name: "imports", rule, table };

const WHY_NAME: &str = "With level 0 and no names, or names of a module that is no package, the interpreter's own __import__ returns the module it finds loaded in sys.modules: it runs no code and writes nothing, and the items it read are what a region is checked on.";
const WHY_FROM: &str = "The name is the item of the exact module's dict, or with no such item and no __getattr__ there the loaded submodule of that name in sys.modules: the items read are what a region is checked on.";

const NAME_READS: &[Read] = &[Read::Global, Read::Modules, Read::Attribute];
// (level, names): an int and None, or an int and a tuple.
const NAME_ROW: Row = Row { operands: &[(0, Class::Group(Group::Immutable)), (1, Class::Group(Group::Immutable))], reads: NAME_READS, writes: Write::None, result: Res::Pinned, enters: Enters::Never, scalars: Scalars::None, why: WHY_NAME, before: Some(import_name) };
const NAMES_ROW: Row = Row { operands: &[(0, Class::Group(Group::Immutable)), (1, Class::Group(Group::ExactTuple))], reads: NAME_READS, writes: Write::None, result: Res::Pinned, enters: Enters::Never, scalars: Scalars::None, why: WHY_NAME, before: Some(import_name) };
const FROM_ROW: Row = Row { operands: &[(0, Class::Group(Group::ExactModule))], reads: &[Read::Attribute, Read::Modules], writes: Write::None, result: Res::Pinned, enters: Enters::Never, scalars: Scalars::None, why: WHY_FROM, before: Some(import_from) };

fn rule(what: &What) -> Taken {
    match *what {
        What::Opcode(IMPORT_NAME, _) => Taken::Rows(&[NAME_ROW, NAMES_ROW]),
        What::Opcode(IMPORT_FROM, _) => Taken::Rows(&[FROM_ROW]),
        _ => Taken::No,
    }
}

fn table() -> Vec<(String, Taken)> {
    [IMPORT_NAME, IMPORT_FROM].iter().map(|op| (opcodes::name(*op).to_string(), rule(&What::Opcode(*op, 0)))).collect()
}

// The names the import machinery asks by, interned once.
struct Names {
    import: usize,
    spec: usize,
    initializing: usize,
    path: usize,
    getattr: usize,
    name: usize,
    // The class importlib makes specs of (importlib.machinery.ModuleSpec),
    // or 0.
    spec_class: usize,
}

unsafe fn names() -> &'static Names {
    static NAMES: OnceLock<Names> = OnceLock::new();
    NAMES.get_or_init(|| {
        let at = |s: &CStr| ffi::PyUnicode_InternFromString(s.as_ptr()) as usize;
        let importlib = reads::dict_get(ffi::PyImport_GetModuleDict(), at(c"_frozen_importlib") as *mut PyObject);
        let spec_class = if exact(importlib, &raw mut ffi::PyModule_Type) { reads::dict_get(ffi::PyModule_GetDict(importlib), at(c"ModuleSpec") as *mut PyObject) } else { std::ptr::null_mut() };
        ffi::Py_XINCREF(spec_class);
        Names { import: at(c"__import__"), spec: at(c"__spec__"), initializing: at(c"_initializing"), path: at(c"__path__"), getattr: at(c"__getattr__"), name: at(c"__name__"), spec_class: spec_class as usize }
    })
}

unsafe fn exact(o: *mut PyObject, tp: *mut ffi::PyTypeObject) -> bool {
    !o.is_null() && ffi::Py_TYPE(o) == tp
}

// Whether `f` is the builtin function __import__ of the builtins module: by
// its C definition, not by where it was found.
unsafe fn own_import(f: *mut PyObject) -> bool {
    if !exact(f, &raw mut ffi::PyCFunction_Type) {
        return false;
    }
    let f = f as *mut ffi::PyCFunctionObject;
    let owner = (*f).m_self;
    if CStr::from_ptr((*(*f).m_ml).ml_name) != c"__import__" || !exact(owner, &raw mut ffi::PyModule_Type) {
        return false;
    }
    let def = ffi::PyModule_GetDef(owner);
    !def.is_null() && CStr::from_ptr((*def).m_name) == c"builtins"
}

// The interpreter's table of loaded modules, if it is an exact dict.
unsafe fn modules() -> Result<*mut PyObject, String> {
    let modules = ffi::PyImport_GetModuleDict();
    if exact(modules, &raw mut ffi::PyDict_Type) { Ok(modules) } else { Err("the table of loaded modules is no exact dict".into()) }
}

// A module the import finds loaded under `key`, with what was read of it.
struct Loaded {
    module: *mut PyObject,
    dict: *mut PyObject,
    spec: *mut PyObject,
    // What the spec has as `_initializing`: False, or NULL for nothing.
    initializing: *mut PyObject,
}

// What the import finds under `key` in `modules` without running anything:
// an exact module that is not being initialized. Err: what it would do
// instead.
unsafe fn loaded(modules: *mut PyObject, key: *mut PyObject) -> Result<Loaded, String> {
    let n = names();
    let module = reads::dict_get(modules, key);
    if module.is_null() || module == ffi::Py_None() {
        return Err(format!("the module {} is not loaded: the import machinery would find it and run its code", reads::text(key)));
    }
    if !exact(module, &raw mut ffi::PyModule_Type) {
        return Err(format!("sys.modules has an object of the class {} as {}, whose attributes are code of that class", reads::type_name(module), reads::text(key)));
    }
    let dict = ffi::PyModule_GetDict(module);
    let spec = if dict.is_null() { std::ptr::null_mut() } else { reads::dict_get(dict, n.spec as *mut PyObject) };
    if spec.is_null() {
        ffi::PyErr_Clear();
        return Err(format!("the module {} has no __spec__ to say whether it is being initialized", reads::text(key)));
    }
    let mut initializing = std::ptr::null_mut();
    if spec != ffi::Py_None() {
        // The import asks the spec for `_initializing` and takes a spec that
        // has no such attribute for one that is not being initialized (a
        // spec importlib.reload made never had it set). Under a class that
        // resolves attributes the generic way the spec says so in its own
        // dict, as False; or it is a spec of importlib's own class, exactly,
        // with no such item in its dict, and the class with its bases has
        // neither the name nor a __getattr__ to be asked for it.
        let mut own = std::ptr::null_mut();
        let name = n.initializing as *mut PyObject;
        let tp = ffi::Py_TYPE(spec);
        let settled = match tr_attr_home(spec, name, &mut own, 0) {
            2 => {
                initializing = reads::dict_get(own, name);
                initializing == ffi::Py_False()
            }
            1 => !own.is_null() && tp as usize == n.spec_class && tr_type_lookup(tp, name).is_null() && tr_type_lookup(tp, n.getattr as *mut PyObject).is_null(),
            _ => false,
        };
        ffi::Py_XDECREF(own);
        if !settled {
            return Err(format!("the module {} is being initialized, or its __spec__ does not say: the import would wait for it", reads::text(key)));
        }
    }
    Ok(Loaded { module, dict, spec, initializing })
}

// The reads of loaded(), as entries of the trace.
unsafe fn found(modules: *mut PyObject, key: *mut PyObject, at: &Loaded) {
    let n = names();
    reads::subscript(modules, key, false);
    reads::attr(at.module, n.spec as *mut PyObject, at.spec, false);
    if at.spec != ffi::Py_None() {
        // In the spec's dict; or absent there, with the spec's class and
        // what that class has for the name, for __getattribute__ and for
        // __getattr__.
        reads::attr(at.spec, n.initializing as *mut PyObject, at.initializing, false);
    }
}

unsafe fn name_of(what: &What, site: &Site) -> Result<*mut PyObject, String> {
    let name = match *what {
        What::Opcode(_, arg) if !site.names.is_null() => *site.names.add(arg as usize),
        _ => std::ptr::null_mut(),
    };
    if exact(name, &raw mut ffi::PyUnicode_Type) { Ok(name) } else { Err("its name is not known".into()) }
}

// IMPORT_NAME(name) on (level, names).
unsafe fn import_name(what: &What, site: &Site) -> Result<(), String> {
    let n = names();
    let name = name_of(what, site)?;
    let &[level, fromlist] = site.operands else { return Err("it has other operands than a level and a list of names".into()) };
    if !exact(level, &raw mut ffi::PyLong_Type) || ffi::PyLong_AsLong(level) != 0 {
        ffi::PyErr_Clear();
        return Err("a relative import finds its package through the globals of the frame".into());
    }
    // None, or a tuple of strs: nothing else is asked of the list then.
    let named = if fromlist == ffi::Py_None() {
        false
    } else if exact(fromlist, &raw mut ffi::PyTuple_Type) && (0..ffi::PyTuple_GET_SIZE(fromlist)).all(|i| exact(ffi::PyTuple_GET_ITEM(fromlist, i), &raw mut ffi::PyUnicode_Type)) {
        ffi::PyTuple_GET_SIZE(fromlist) > 0
    } else {
        return Err("its list of names is neither None nor a tuple of strs".into());
    };
    let builtins = tr_frame_builtins(site.frame);
    if !exact(builtins, &raw mut ffi::PyDict_Type) || !own_import(reads::dict_get(builtins, n.import as *mut PyObject)) {
        return Err("the builtins of the frame have an __import__ that is not the interpreter's own".into());
    }
    let modules = modules()?;
    let module = loaded(modules, name)?;
    // `import a.b` gives a: the module loaded under the name up to its
    // first dot, found the same way.
    let dot = if named { -1 } else { ffi::PyUnicode_FindChar(name, '.' as u32, 0, ffi::PyUnicode_GetLength(name), 1) };
    let top = if dot > 0 { ffi::PyUnicode_Substring(name, 0, dot) } else { std::ptr::null_mut() };
    if dot >= 0 && top.is_null() {
        ffi::PyErr_Clear();
        return Err("its name has no package in front of the first dot".into());
    }
    let package = if top.is_null() { Ok(None) } else { loaded(modules, top).map(Some) };
    // Names of a package go through importlib; a module that is none is
    // asked for __path__, which its own __getattr__ would answer.
    let asked = [n.path, n.getattr].map(|k| k as *mut PyObject);
    let plain = !named || asked.iter().all(|k| reads::dict_get(module.dict, *k).is_null());
    let ok = match package {
        Err(why) => Err(why),
        Ok(_) if !plain => Err(format!("it takes names from {}, a package or a module with a __getattr__: importlib's Python code, or that function, runs", reads::text(name))),
        Ok(package) => {
            reads::subscript(builtins, n.import as *mut PyObject, false);
            found(modules, name, &module);
            if let Some(package) = package {
                found(modules, top, &package);
            }
            if named {
                asked.iter().for_each(|k| reads::attr(module.module, *k, std::ptr::null_mut(), false));
            }
            Ok(())
        }
    };
    ffi::Py_XDECREF(top);
    ok
}

// IMPORT_FROM(name) on an exact module.
unsafe fn import_from(what: &What, site: &Site) -> Result<(), String> {
    let n = names();
    let name = name_of(what, site)?;
    let &[module] = site.operands else { return Err("it has other operands than a module".into()) };
    let dict = ffi::PyModule_GetDict(module);
    // What the class of modules has under the name comes before the dict
    // (`__dict__`, `__class__`) or after it (`__repr__`): neither is a row.
    if dict.is_null() || !tr_type_lookup(&raw mut ffi::PyModule_Type, name).is_null() {
        ffi::PyErr_Clear();
        return Err(format!("the class of modules has an attribute {} of its own", reads::text(name)));
    }
    let value = reads::dict_get(dict, name);
    if !value.is_null() {
        reads::attr(module, name, value, false);
        return Ok(());
    }
    if !reads::dict_get(dict, n.getattr as *mut PyObject).is_null() {
        return Err(format!("the module has no {} and a __getattr__, which is asked for it", reads::text(name)));
    }
    // The way out for a circular import: the submodule in sys.modules.
    let package = reads::dict_get(dict, n.name as *mut PyObject);
    let full = if exact(package, &raw mut ffi::PyUnicode_Type) { ffi::PyUnicode_FromFormat(c"%U.%U".as_ptr(), package, name) } else { std::ptr::null_mut() };
    if full.is_null() {
        ffi::PyErr_Clear();
        return Err(format!("the module has no {} and no name to look a submodule up by", reads::text(name)));
    }
    let ok = modules().and_then(|modules| {
        let submodule = loaded(modules, full)?;
        for k in [name, n.getattr as *mut PyObject] {
            reads::attr(module, k, std::ptr::null_mut(), false);
        }
        reads::attr(module, n.name as *mut PyObject, package, false);
        found(modules, full, &submodule);
        Ok(())
    });
    ffi::Py_DECREF(full);
    ok
}
