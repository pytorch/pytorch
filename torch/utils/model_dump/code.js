import { h, Component, render } from 'https://unpkg.com/preact?module';
import htm from 'https://unpkg.com/htm?module';

const html = htm.bind(h);

const BURNED_IN_MODEL_INFO = null;

// https://stackoverflow.com/a/20732091
function humanFileSize(size) {
  if (size == 0) { return "0 B"; }
  var i = Math.floor( Math.log(size) / Math.log(1024) );
  return (size / Math.pow(1024, i)).toFixed(2) * 1 + ' ' + ['B', 'kB', 'MB', 'GB', 'TB'][i];
}

function caret(down) {
  return down ? "\u25BE" : "\u25B8";
}

class Blamer {
  constructor() {
    this.blame_on_click = false;
    this.aux_content_pane = null;
  }

  setAuxContentPane(pane) {
    this.aux_content_pane = pane;
  }

  readyBlame() {
    this.blame_on_click = true;
  }

  maybeBlame(arg) {
    if (!this.blame_on_click) {
      return;
    }
    this.blame_on_click = false;
    if (!this.aux_content_pane) {
      return;
    }
    this.aux_content_pane.doBlame(arg);
  }
}

let blame = new Blamer();

class Hider extends Component {
  constructor() {
    super();
    this.state = { shown: null };
  }

  componentDidMount() {
    this.setState({ shown: this.props.shown === "true" });
  }

  render({name, children}, {shown}) {
    let my_caret = html`<span class=caret onClick=${() => this.click()} >${caret(shown)}</span>`;
    return html`<div data-hider-title=${name} data-shown=${shown}>
      <h2>${my_caret} ${name}</h2>
      <div>${shown ? this.props.children : []}</div></div>`;
  }

  click() {
    this.setState({shown: !this.state.shown});
  }
}

function ModelSizeSection({model: {file_size, zip_files}}) {
  let store_size = 0;
  let compr_size = 0;
  for (const zi of zip_files) {
    if (zi.compression === 0) {
      // TODO: Maybe check that compressed_size === file_size.
      store_size += zi.compressed_size;
    } else {
      compr_size += zi.compressed_size;
    }
  }
  let zip_overhead = file_size - store_size - compr_size;
  // TODO: Better formatting.  Right-align this.
  return html`
    <${Hider} name="Model Size" shown=true>
    <pre>.
      Model size: ${file_size} (${humanFileSize(file_size)})
      Stored files: ${store_size} (${humanFileSize(store_size)})
      Compressed files: ${compr_size} (${humanFileSize(compr_size)})
      Zip overhead: ${zip_overhead} (${humanFileSize(zip_overhead)})
    </pre><//>`;
}

function StructuredDataSection({name, data, shown}) {
  return html`
    <${Hider} name=${name} shown=${shown}>
    <div style="font-family:monospace;">
      <${StructuredData} data=${data} indent="" prefix=""/>
    </div><//>`;
}

class StructuredData extends Component {
  constructor() {
    super();
    this.state = { shown: false };

    this.INLINE_TYPES = new Set(["boolean", "number", "string"])
    this.IGNORED_STATE_KEYS = new Set(["training", "_is_full_backward_hook"])
  }

  click() {
    this.setState({shown: !this.state.shown});
  }

  expando(data) {
    if (data === null || this.INLINE_TYPES.has(typeof(data))) {
      return false;
    }
    if (typeof(data) != "object") {
      throw new Error("Not an object");
    }
    if (Array.isArray(data)) {
      // TODO: Maybe show simple lists and tuples on one line.
      return true;
    }
    if (data.__tuple_values__) {
      // TODO: Maybe show simple lists and tuples on one line.
      return true;
    }
    if (data.__is_dict__) {
      // TODO: Maybe show simple (empty?) dicts on one line.
      return true;
    }
    if (data.__module_type__) {
      return true;
    }
    if (data.__tensor_v2__) {
      return false;
    }
    if (data.__qtensor__) {
      return false;
    }
    throw new Error("Can't handle data type.", data);
  }

  renderHeadline(data) {
    if (data === null) {
      return "None";
    }
    if (typeof(data) == "boolean") {
      const sd = String(data);
      return sd.charAt(0).toUpperCase() + sd.slice(1);
    }
    if (typeof(data) == "number") {
      return JSON.stringify(data);
    }
    if (typeof(data) == "string") {
      return JSON.stringify(data);
    }
    if (typeof(data) != "object") {
      throw new Error("Not an object");
    }
    if (Array.isArray(data)) {
      return "list([";
    }
    if (data.__tuple_values__) {
      return "tuple((";
    }
    if (data.__is_dict__) {
      return "dict({";
    }
    if (data.__module_type__) {
      return data.__module_type__ + "()";
    }
    if (data.__tensor_v2__) {
      const [storage, offset, size, stride, grad] = data.__tensor_v2__;
      const [dtype, key, device, numel] = storage;
      return this.renderTensor(
        "tensor", dtype, key, device, numel, offset, size, stride, grad, []);
    }
    if (data.__qtensor__) {
      const [storage, offset, size, stride, quantizer, grad] = data.__qtensor__;
      const [dtype, key, device, numel] = storage;
      let extra_parts = [];
      if (quantizer[0] == "per_tensor_affine") {
        extra_parts.push(`scale=${quantizer[1]}`);
        extra_parts.push(`zero_point=${quantizer[2]}`);
      } else {
        extra_parts.push(`quantizer=${quantizer[0]}`);
      }
      return this.renderTensor(
        "qtensor", dtype, key, device, numel, offset, size, stride, grad, extra_parts);
    }
    throw new Error("Can't handle data type.", data);
  }

  renderTensor(
      prefix,
      dtype,
      storage_key,
      device,
      storage_numel,
      offset,
      size,
      stride,
      grad,
      extra_parts) {
    let parts = [
      "(" + size.join(",") + ")",
      dtype,
    ];
    parts.push(...extra_parts);
    if (device != "cpu") {
      parts.push(device);
    }
    if (grad) {
      parts.push("grad");
    }
    // TODO: Check stride and indicate if the tensor is channels-last or non-contiguous
    // TODO: Check size, stride, offset, and numel and indicate if
    // the tensor doesn't use all data in storage.
    // TODO: Maybe show key?
    void(offset);
    void(stride);
    void(storage_key);
    void(storage_numel);
    return prefix + "(" + parts.join(", ") + ")";
  }

  renderBody(indent, data) {
    if (data === null || this.INLINE_TYPES.has(typeof(data))) {
      throw "Should not reach here."
    }
    if (typeof(data) != "object") {
      throw new Error("Not an object");
    }
    if (Array.isArray(data)) {
      let new_indent = indent + "\u00A0\u00A0";
      let parts = [];
      for (let idx = 0; idx < data.length; idx++) {
        // Does it make sense to put explicit index numbers here?
        parts.push(html`<br/><${StructuredData} prefix=${idx + ": "} indent=${new_indent} data=${data[idx]} />`);
      }
      return parts;
    }
    if (data.__tuple_values__) {
      // Handled the same as lists.
      return this.renderBody(indent, data.__tuple_values__);
    }
    if (data.__is_dict__) {
      let new_indent = indent + "\u00A0\u00A0";
      let parts = [];
      for (let idx = 0; idx < data.keys.length; idx++) {
        if (typeof(data.keys[idx]) != "string") {
          parts.push(html`<br/>${new_indent}Non-string key`);
        } else {
          parts.push(html`<br/><${StructuredData} prefix=${data.keys[idx] + ": "} indent=${new_indent} data=${data.values[idx]} />`);
        }
      }
      return parts;
    }
    if (data.__module_type__) {
      const mstate = data.state;
      if (mstate === null || typeof(mstate) != "object") {
        throw new Error("Bad module state");
      }
      let new_indent = indent + "\u00A0\u00A0";
      let parts = [];
      if (mstate.__is_dict__) {
        // TODO: Less copy/paste between this and normal dicts.
        for (let idx = 0; idx < mstate.keys.length; idx++) {
          if (typeof(mstate.keys[idx]) != "string") {
            parts.push(html`<br/>${new_indent}Non-string key`);
          } else if (this.IGNORED_STATE_KEYS.has(mstate.keys[idx])) {
            // Do nothing.
          } else {
            parts.push(html`<br/><${StructuredData} prefix=${mstate.keys[idx] + ": "} indent=${new_indent} data=${mstate.values[idx]} />`);
          }
        }
      } else if (mstate.__tuple_values__) {
        parts.push(html`<br/><${StructuredData} prefix="" indent=${new_indent} data=${mstate} />`);
      } else if (mstate.__module_type__) {
        // We normally wouldn't have the state of a module be another module,
        // but we use "modules" to encode special values (like Unicode decode
        // errors) that might be valid states.  Just go with it.
        parts.push(html`<br/><${StructuredData} prefix="" indent=${new_indent} data=${mstate} />`);
      } else {
        throw new Error("Bad module state");
      }
      return parts;
    }
    if (data.__tensor_v2__) {
      throw "Should not reach here."
    }
    if (data.__qtensor__) {
      throw "Should not reach here."
    }
    throw new Error("Can't handle data type.", data);
  }

  render({data, indent, prefix}, {shown}) {
    const exp = this.expando(data) ? html`<span class=caret onClick=${() => this.click()} >${caret(shown)} </span>` : "";
    const headline = this.renderHeadline(data);
    const body = shown ? this.renderBody(indent, data) : "";
    return html`${indent}${exp}${prefix}${headline}${body}`;
  }
}

function ZipContentsSection({model: {zip_files}}) {
  // TODO: Add human-readable sizes?
  // TODO: Add sorting options?
  // TODO: Add hierarchical collapsible tree?
  return html`
    <${Hider} name="Zip Contents" shown=false>
    <table>
      <thead>
        <tr>
          <th>Mode</th>
          <th>Size</th>
          <th>Compressed</th>
          <th>Name</th>
        </tr>
      </thead>
      <tbody style="font-family:monospace;">
        ${zip_files.map(zf => html`<tr>
          <td>${{0: "store", 8: "deflate"}[zf.compression] || zf.compression}</td>
          <td>${zf.file_size}</td>
          <td>${zf.compressed_size}</td>
          <td>${zf.filename}</td>
        </tr>`)}
      </tbody>
    </table><//>`;
}

function CodeSection({model: {code_files}}) {
  return html`
    <${Hider} name="Code" shown=false>
    <div>
      ${Object.entries(code_files).map(([fn, code]) => html`<${OneCodeSection}
          filename=${fn} code=${code} />`)}
    </div><//>`;
}

class OneCodeSection extends Component {
  constructor() {
    super();
    this.state = { shown: false };
  }

  click() {
    const shown = !this.state.shown;
    this.setState({shown: shown});
  }

  render({filename, code}, {shown}) {
    const header = html`
        <h3 style="font-family:monospace;">
        <span class=caret onClick=${() => this.click()} >${caret(shown)} </span>
        ${filename}</h3>
        `;
    if (!shown) {
      return header;
    }
    return html`
      ${header}
      <pre>${code.map(c => this.renderBlock(c))}</pre>
      `;
  }

  renderBlock([text, ist_file, line, ist_s_text, s_start, s_end]) {
    return html`<span
        onClick=${() => blame.maybeBlame({ist_file, line, ist_s_text, s_start, s_end})}
      >${text}</span>`;
  }
}

function ExtraJsonSection({files}) {
  return html`
    <${Hider} name="Extra files (JSON)" shown=false>
    <div>
      <p>Use "Log Raw Model Info" for hierarchical view in browser console.</p>
      ${Object.entries(files).map(([fn, json]) => html`<${OneJsonSection}
          filename=${fn} json=${json} />`)}
    </div><//>`;
}

class OneJsonSection extends Component {
  constructor() {
    super();
    this.state = { shown: false };
  }

  click() {
    const shown = !this.state.shown;
    this.setState({shown: shown});
  }

  render({filename, json}, {shown}) {
    const header = html`
        <h3 style="font-family:monospace;">
        <span class=caret onClick=${() => this.click()} >${caret(shown)} </span>
        ${filename}</h3>
        `;
    if (!shown) {
      return header;
    }
    return html`
      ${header}
      <pre>${JSON.stringify(json, null, 2)}</pre>
      `;
  }
}

function ExtraPicklesSection({files}) {
  return html`
    <${Hider} name="Extra Pickles" shown=false>
    <div>
      ${Object.entries(files).map(([fn, content]) => html`<${OnePickleSection}
          filename=${fn} content=${content} />`)}
    </div><//>`;
}

class OnePickleSection extends Component {
  constructor() {
    super();
    this.state = { shown: false };
  }

  click() {
    const shown = !this.state.shown;
    this.setState({shown: shown});
  }

  render({filename, content}, {shown}) {
    const header = html`
        <h3 style="font-family:monospace;">
        <span class=caret onClick=${() => this.click()} >${caret(shown)} </span>
        ${filename}</h3>
        `;
    if (!shown) {
      return header;
    }
    return html`
      ${header}
      <pre>${content}</pre>
      `;
  }
}

function assertStorageAreEqual(key, lhs, rhs) {
  if (lhs.length !== rhs.length ||
    !lhs.every((val, idx) => val === rhs[idx])) {
    throw new Error("Storage mismatch for key '" + key + "'");
  }
}

function computeTensorMemory(numel, dtype) {
  const sizes = {
    "Byte": 1,
    "Char": 1,
    "Short": 2,
    "Int": 4,
    "Long": 8,
    "Half": 2,
    "Float": 4,
    "Double": 8,
    "ComplexHalf": 4,
    "ComplexFloat": 8,
    "ComplexDouble": 16,
    "Bool": 1,
    "QInt8": 1,
    "QUInt8": 1,
    "QInt32": 4,
    "BFloat16": 2,
  };
  let dtsize = sizes[dtype];
  if (!dtsize) {
    throw new Error("Unrecognized dtype: " + dtype);
  }
  return numel * dtsize;
}

// TODO: Maybe track by dtype as well.
// TODO: Maybe distinguish between visible size and storage size.
function getTensorStorages(data) {
  if (data === null) {
    return new Map();
  }
  if (typeof(data) == "boolean") {
    return new Map();
  }
  if (typeof(data) == "number") {
    return new Map();
  }
  if (typeof(data) == "string") {
    return new Map();
  }
  if (typeof(data) != "object") {
    throw new Error("Not an object");
  }
  if (Array.isArray(data)) {
    let result = new Map();
    for (const item of data) {
      const tensors = getTensorStorages(item);
      for (const [key, storage] of tensors.entries()) {
        if (!result.has(key)) {
          result.set(key, storage);
        } else {
          const old_storage = result.get(key);
          assertStorageAreEqual(key, old_storage, storage);
        }
      }
    }
    return result;
  }
  if (data.__tuple_values__) {
    return getTensorStorages(data.__tuple_values__);
  }
  if (data.__is_dict__) {
    return getTensorStorages(data.values);
  }
  if (data.__module_type__) {
    return getTensorStorages(data.state);
  }
  if (data.__tensor_v2__) {
    const [storage, offset, size, stride, grad] = data.__tensor_v2__;
    const [dtype, key, device, numel] = storage;
    return new Map([[key, storage]]);
  }
  if (data.__qtensor__) {
    const [storage, offset, size, stride, quantizer, grad] = data.__qtensor__;
    const [dtype, key, device, numel] = storage;
    return new Map([[key, storage]]);
  }
  throw new Error("Can't handle data type.", data);
}

function getTensorMemoryByDevice(pickles) {
  let all_tensors = [];
  for (const [name, pickle] of pickles) {
    const tensors = getTensorStorages(pickle);
    all_tensors.push(...tensors.values());
  }
  let result = {};
  for (const storage of all_tensors.values()) {
    const [dtype, key, device, numel] = storage;
    const size = computeTensorMemory(numel, dtype);
    result[device] = (result[device] || 0) + size;
  }
  return result;
}

// Make this a separate component so it is rendered lazily.
class OpenTensorMemorySection extends Component {
  render({model: {model_data, constants}}) {
    let sizes = getTensorMemoryByDevice(new Map([
      ["data", model_data],
      ["constants", constants],
    ]));
    return html`
      <table>
        <thead>
          <tr>
            <th>Device</th>
            <th>Bytes</th>
            <th>Human</th>
          </tr>
        </thead>
        <tbody style="font-family:monospace;">
          ${Object.entries(sizes).map(([dev, size]) => html`<tr>
            <td>${dev}</td>
            <td>${size}</td>
            <td>${humanFileSize(size)}</td>
          </tr>`)}
        </tbody>
      </table>`;
  }
}

function TensorMemorySection({model}) {
  return html`
    <${Hider} name="Tensor Memory" shown=false>
    <${OpenTensorMemorySection} model=${model} /><//>`;
}

class AuxContentPane extends Component {
  constructor() {
    super();
    this.state = {
      blame_info: null,
    };
  }

  doBlame(arg) {
    this.setState({...this.state, blame_info: arg});
  }

  render({model: {interned_strings}}, {blame_info}) {
    let blame_content = "";
    if (blame_info) {
      const {ist_file, line, ist_s_text, s_start, s_end} = blame_info;
      let s_text = interned_strings[ist_s_text];
      if (s_start != 0 || s_end != s_text.length) {
        let prefix = s_text.slice(0, s_start);
        let main = s_text.slice(s_start, s_end);
        let suffix = s_text.slice(s_end);
        s_text = html`${prefix}<strong>${main}</strong>${suffix}`;
      }
      blame_content = html`
        <h3>${interned_strings[ist_file]}:${line}</h3>
        <pre>${s_start}:${s_end}</pre>
        <pre>${s_text}</pre><br/>
        `;
    }
    return html`
      <button onClick=${() => blame.readyBlame()}>Blame Code</button>
      <br/>
      ${blame_content}
      `;
  }
}

class App extends Component {
  constructor() {
    super();
    this.state = {
      err: false,
      model: null,
    };
  }

  componentDidMount() {
    const app = this;
    if (BURNED_IN_MODEL_INFO !== null) {
      app.setState({model: BURNED_IN_MODEL_INFO});
    } else {
      fetch("./model_info.json").then(function(response) {
        if (!response.ok) {
          throw new Error("Response not ok.");
        }
        return response.json();
      }).then(function(body) {
        app.setState({model: body});
      }).catch(function(error) {
        console.log("Top-level error: ", error);
      });
    }
  }

  componentDidCatch(error) {
    void(error);
    this.setState({...this.state, err: true});
  }

  render(_, {err}) {
    if (this.state.model === null) {
      return html`<h1>Loading...</h1>`;
    }

    const model = this.state.model.model;

    let error_msg = "";
    if (err) {
      error_msg = html`<h2 style="background:red">An error occurred.  Check console</h2>`;
    }

    return html`
      ${error_msg}
      <div id=main_content style="position:absolute;width:99%;height:79%;overflow:scroll">
        <h1>TorchScript Model (version ${model.version}): ${model.title}</h1>
        <button onClick=${() => console.log(model)}>Log Raw Model Info</button>
        <${ModelSizeSection} model=${model}/>
        <${StructuredDataSection} name="Model Data" data=${model.model_data} shown=true/>
        <${StructuredDataSection} name="Constants" data=${model.constants} shown=false/>
        <${ZipContentsSection} model=${model}/>
        <${CodeSection} model=${model}/>
        <${ExtraJsonSection} files=${model.extra_files_jsons}/>
        <${ExtraPicklesSection} files=${model.extra_pickles}/>
        <${TensorMemorySection} model=${model}/>
      </div>
      <div id=aux_content style="position:absolute;width:99%;top:80%;height:20%;overflow:scroll">
        <${AuxContentPane}
          err=${this.state.error}
          model=${model}
          ref=${(p) => blame.setAuxContentPane(p)}/>
      </div>
      `;
  }
}

render(h(App), document.body);
