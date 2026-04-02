document.addEventListener('DOMContentLoaded', function() {
    jsonData = null

    // build module level tree
    function buildTree(data, level = 0) {
        let html = `<ul class = "tree">`;

        // build module node
        data.forEach(node => {
            // check json forward pass information
            html += `<li><span class="caret"><b>${node.fqn}</b></span>`

            if (node.module_type.length > 0 || node.parameters.length > 0) {

                html += `<br><span class="forward-pass"><span class="caret" id="${node.fqn}-module-information-caret">Module Information</span></span></br>`

                html += `<div style = "display: none;" id="${node.fqn}-module-information">`
                html += `<span class="forward-pass">`

                html += `<span style="margin-left: 20px;">module type: ${node.module_type}</span>`

                if (node.parameters.length > 0){
                    html += `<table style="margin-left: 60px;">`
                    html += `<tr>`
                    html += `<th>Parameter</th>`
                    html += `<th>Sharding</th>`

                    html += `</tr>`

                    for (parameter_info of node.parameters) {
                        html += `<tr>`
                        html += `<td> ${parameter_info[0]} </td>`

                        html += `<td> ${parameter_info[1]} </td>`

                        html += `</tr>`
                    }
                    html += `</table>`
                }
                html += `</span>`
                html += `</div>`

            }

            if (node.collectives_forward.length > 0 || node.operations_forward.length > 0) {
                html += `<br><span class="forward-pass">Forward Pass</span><br>`
                // build forward module collective count table
                if (node.collectives_forward.length > 0){
                    html += `<span class="forward-pass">`
                    html += `<table>`
                    html += `<tr>`
                    html += `<th>Collective</th>`
                    html += `<th>Count</th>`
                    html += `</tr>`
                    for (collective_entry of node.collectives_forward){
                        html += `<tr>`
                        html += `<td> ${collective_entry[0]} </td>`
                        html += `<td> ${collective_entry[1]} </td>`
                        html += `</tr>`
                    }
                    html += `</table>`
                    html += `</span>`
                }

                // build forward module operation table
                if(node.operations_forward.length > 0){
                    html += `<span class="forward-pass">`
                    html += `<span class="caret" id="${node.fqn}-forward-operations-caret">Operations</span>`
                    html += `<span class="forward-pass">`
                    html += `<table style="margin-left: 60px; display: none;" id="${node.fqn}-forward-operations-list">`
                    html += `<tr>`
                    html += `<th>Operation</th>`
                    html += `<th>Input Shape</th>`
                    html += `<th>Input Sharding</th>`
                    html += `<th>Device Mesh</th>`
                    html += `</tr>`

                    for (operation_entry of node.operations_forward) {
                        html += `<tr>`
                        html += `<td> ${operation_entry.name} </td>`

                        html += `<td>`
                        for (shape of operation_entry.input_shape) {
                            html += `<p> ${shape} </p>`
                        }
                        html += `</td>`

                        html += `<td>`
                        for (sharding of operation_entry.input_sharding) {
                            html += `<p> ${sharding} </p>`
                        }
                        html += `</td>`
                        html += `<td> ${operation_entry.device_mesh} </td>`

                        html += `</tr>`
                    }
                    html += `</table>`
                    html += `</span>`
                    html += `</span>`
                }
            }


            // check json backward pass information
            if (node.collectives_backward.length > 0 || node.operations_backward.length > 0) {
                html += `<br><span class="forward-pass">Backward Pass</span><br>`

                // build backward module collective count table
                if (node.collectives_backward.length > 0){
                    html += `<span class="forward-pass">`
                    html += `<table>`
                    html += `<tr>`
                    html += `<th>Collective</th>`
                    html += `<th>Count</th>`
                    html += `</tr>`
                    for (collective_entry of node.collectives_backward){
                        html += `<tr>`
                        html += `<td> ${collective_entry[0]} </td>`
                        html += `<td> ${collective_entry[1]} </td>`
                        html += `</tr>`
                    }
                    html += `</table>`
                    html += `</span>`
                }

                // build forward module operation table
                if(node.operations_backward.length > 0){
                    html += `<span class="forward-pass">`
                    html += `<span class="caret" id="${node.fqn}-backward-operations-caret">Operations</span>`
                    html += `<span class="forward-pass">`
                    html += `<table style="margin-left: 60px; display: none;" id="${node.fqn}-backward-operations-list">`
                    html += `<tr>`
                    html += `<th>Operation</th>`
                    html += `<th>Input Shape</th>`
                    html += `<th>Input Sharding</th>`
                    html += `<th>Device Mesh</th>`
                    html += `</tr>`


                    for (operation_entry of node.operations_backward) {
                        html += `<tr>`
                        html += `<td> ${operation_entry.name} </td>`

                        html += `<td>`
                        for (shape of operation_entry.input_shape) {
                            html += `<p> ${shape} </p>`
                        }
                        html += `</td>`

                        html += `<td>`
                        for (sharding of operation_entry.input_sharding) {
                            html += `<p> ${sharding} </p>`
                        }
                        html += `</td>`
                        html += `<td> ${operation_entry.device_mesh} </td>`
                        html += `</tr>`
                    }

                    html += `</table>`
                    html += `</span>`
                    html += `</span>`
                }
            }

            // check json activation checkpointing information
            if(node.operations_checkpointing.length > 0){
                html += `<br><span class="forward-pass">Activation Checkpointing</span><br>`
                html += `<span class="forward-pass">`
                html += `<span class="caret" id="${node.fqn}-checkpointing-operations-caret">Operations</span>`
                html += `<span class="forward-pass">`
                html += `<table style="margin-left: 60px; display: none;" id="${node.fqn}-checkpointing-operations-list">`
                html += `<tr>`
                html += `<th>Operation</th>`
                html += `<th>Input Shape</th>`
                html += `<th>Input Sharding</th>`
                html += `<th>Device Mesh</th>`
                html += `</tr>`


                for (operation_entry of node.operations_checkpointing) {
                    html += `<tr>`
                    html += `<td> ${operation_entry.name} </td>`

                    html += `<td>`
                    for (shape of operation_entry.input_shape) {
                        html += `<p> ${shape} </p>`
                    }
                    html += `</td>`

                    html += `<td>`
                    for (sharding of operation_entry.input_sharding) {
                        html += `<p> ${sharding} </p>`
                    }
                    html += `</td>`
                    html += `<td> ${operation_entry.device_mesh} </td>`
                    html += `</tr>`
                }

                html += `</table>`
                html += `</span>`
                html += `</span>`
            }

            // checks for sub-modules and builds tree recursively
            if (node.children.length > 0) {
                html += `<ul class = "nested">`;
                html += buildTree(node.children, level + 1);
                html += `</ul>`;
            }
            html += `</li>`;
        });
        html += `</ul>`;
        return html
    }


    const container = document.getElementById('tree-container');
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');

    // controls operation table visibility
    document.addEventListener('click', function(event) {
        if (event.target.id.includes('forward-operations-caret')) {
            const fqn = event.target.id.split('-')[0];
            const operationsListId = `${fqn}-forward-operations-list`;
            const operationsList = document.getElementById(operationsListId);
            if (operationsList) {
                operationsList.style.display = operationsList.style.display === 'none' ? 'block' : 'none';
            }
        }
        if (event.target.id.includes('backward-operations-caret')) {
            const fqn = event.target.id.split('-')[0];
            const operationsListId = `${fqn}-backward-operations-list`;
            const operationsList = document.getElementById(operationsListId);
            if (operationsList) {
                operationsList.style.display = operationsList.style.display === 'none' ? 'block' : 'none';
            }
        }
        if (event.target.id.includes('checkpointing-operations-caret')) {
            const fqn = event.target.id.split('-')[0];
            const operationsListId = `${fqn}-checkpointing-operations-list`;
            const operationsList = document.getElementById(operationsListId);
            if (operationsList) {
                operationsList.style.display = operationsList.style.display === 'none' ? 'block' : 'none';
            }
        }
        if (event.target.id.includes('module-information-caret')) {
            const fqn = event.target.id.split('-')[0];
            const operationsListId = `${fqn}-module-information`;
            const operationsList = document.getElementById(operationsListId);
            if (operationsList) {
                operationsList.style.display = operationsList.style.display === 'none' ? 'block' : 'none';
            }
        }
    });

    // controls module visibility
    container.addEventListener('click', function(event) {
        const target = event.target

        if (target && target.classList.contains('caret')) {
            target.classList.toggle('caret-down');
            const nestedList = target.parentElement.querySelector('.nested');
            if (nestedList) {
                nestedList.classList.toggle('active')
            }
        }
    });


    // allows user to choose file via drag and drop
    dropArea.addEventListener('drop', (event) => {
        event.preventDefault();

        const files = event.dataTransfer.files;
        const file = files[0];
        // Read the contents of the JSON file
        const reader = new FileReader();
        reader.onload = () => {
            const jsonData = JSON.parse(reader.result);
            container.innerHTML = buildTree([jsonData]);
        };
        reader.readAsText(file);

    });

    // allows user to choose file via file selector
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        // Read the contents of the JSON file
        const reader = new FileReader();
        reader.onload = () => {
            const jsonData = JSON.parse(reader.result);
            container.innerHTML = buildTree([jsonData]);
        };
        reader.readAsText(file);
    });

    // prevents users from dragging and dropping file outside selection zone and opening file in new tab
    document.addEventListener('drop', function(event) {
        event.preventDefault();
    });
});
