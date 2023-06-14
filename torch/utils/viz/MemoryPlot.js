import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.7.0/+esm";
import {schemeTableau10} from "https://cdn.skypack.dev/d3-scale-chromatic@3";
import {axisLeft} from "https://cdn.skypack.dev/d3-axis@3";
import {scaleLinear} from "https://cdn.skypack.dev/d3-scale@4";
import {zoom, zoomIdentity} from "https://cdn.skypack.dev/d3-zoom@3";
import {brushX} from "https://cdn.skypack.dev/d3-brush@3";

let alloc_data = null

function process_alloc_data(max_entries) {
    let current = []
    let current_data = []
    let data = []
    let max_size = 0

    let total_mem = 0
    let total_summarized_mem = 0
    let timestep = 0

    let max_at_time = []


    let summarized_mem = {elem: 'summarized', timesteps: [], offsets: [total_mem], size: [], color: 0}
    let summarized_elems = {}

    function advance(n) {
        summarized_mem.timesteps.push(timestep)
        summarized_mem.offsets.push(total_mem)
        summarized_mem.size.push(total_summarized_mem)
        timestep += n
        for (let i = 0; i < n; i++) {
            max_at_time.push(total_mem + total_summarized_mem)
        }
    }

    let sizes = alloc_data.elements_size.map((x, i) => [x, i]).sort(([x, xi], [y, yi]) => y - x)

    let draw_elem = {}
    for (const [s, e] of sizes.slice(0, max_entries)) {
        draw_elem[e] = true
    }

    function add_allocation(elem) {
        let size = alloc_data.elements_size[elem]
        current.push(elem)
        let color = elem
        if (alloc_data.elements_category !== null) {
            color = alloc_data.elements_category[elem]
        }
        let e = {elem: elem, timesteps: [timestep], offsets: [total_mem], size: size, color: color}
        current_data.push(e)
        data.push(e)
        total_mem += size
    }

    for (const elem of alloc_data.initially_allocated) {
        if (elem in draw_elem) {
            add_allocation(elem)
        } else {
            total_summarized_mem += alloc_data.elements_size[elem]
            summarized_elems[elem] = true
        }
    }

    for (const action of alloc_data.actions) {
        const elem = action
        const size = alloc_data.elements_size[elem]
        if ( !(elem in draw_elem)) {
            if (elem in summarized_elems) {
                advance(1)
                total_summarized_mem -= size
                summarized_elems[elem] = null
            } else {
                total_summarized_mem += size
                summarized_elems[elem] = true
                advance(1)
            }
            continue
        }
        const idx = current.findLastIndex(x => x === elem)
        // first time we see an action we add it
        // second time we remove it
        if (idx == -1) {
            add_allocation(elem)
            advance(1)
        } else {
            advance(1)
            const removed = current_data[idx]
            removed.timesteps.push(timestep)
            removed.offsets.push(removed.offsets.at(-1))
            current.splice(idx, 1)
            current_data.splice(idx, 1)

            if (idx < current.length) {
                for (let j = idx; j < current.length; j++) {
                    const e = current_data[j]
                    e.timesteps.push(timestep)
                    e.offsets.push(e.offsets.at(-1))
                    e.timesteps.push(timestep + 3)
                    e.offsets.push(e.offsets.at(-1) - size)
                }
                advance(3)
            }
            total_mem -= size
        }
        max_size = Math.max(total_mem + total_summarized_mem, max_size)
    }

    for (const elem of current_data) {
        elem.timesteps.push(timestep)
        elem.offsets.push(elem.offsets.at(-1))
    }
    data.push(summarized_mem)

    return {
        max_size: max_size,
        allocations_over_time: data,
        max_at_time: max_at_time,
        summarized_mem: summarized_mem,
        context_for_id:  (elem) => {
            let strings = []
            let id = alloc_data.elements_info[elem]
            while (id !== null) {
                const [sid, next_id] = alloc_data.suffix_table[id]
                strings.push(alloc_data.string_table[sid])
                id = next_id
            }
            return `${strings.join('\n')}\n`
        }
    }
}

function MemoryPlot(svg, data, left_pad, colors=schemeTableau10) {
    function format_points(d) {
        const size = d.size
        const xs = d.timesteps.map(t => xscale(t))
        const bottom = d.offsets.map(t => yscale(t))
        const m = Array.isArray(size) ? ((t, i) => yscale(t + size[i]))
                                      :  (t => yscale(t + size))
        const top = d.offsets.map(m)
        const p0 = xs.map((x, i) => `${x},${bottom[i]}`)
        const p1 = xs.map((x, i) => `${x},${top[i]}`).reverse()

        return `${p0.join(' ')} ${p1.join(' ')}`
    }

    let max_timestep = data.max_at_time.length
    let max_size = data.max_size

    let width = svg.attr('width')
    let height = svg.attr('height')
    let plot_width = width - left_pad
    let plot_height = height

    let yscale = scaleLinear().domain([0, max_size]).range([plot_height, 0]);
    let heightscale = scaleLinear().domain([0, max_size]).range([0, plot_height]);
    let yaxis = axisLeft(yscale).tickFormat(d3.format("~s"))
    let xscale = scaleLinear().domain([0, max_timestep]).range([0, plot_width])
    let plot_coordinate_space = svg.append("g").attr("transform", `translate(${left_pad}, ${0})`)
    let plot_outer = plot_coordinate_space.append('g')

    function view_rect(a) {
        return a.append('rect').attr('x', 0).attr('y', 0)
                .attr('width', plot_width).attr('height', plot_height)
                .attr('fill', 'white')
    }

    view_rect(plot_outer)

    let cp = svg.append("clipPath").attr("id", "clip")
    view_rect(cp)
    plot_outer.attr('clip-path', "url(#clip)")


    let zoom_group = plot_outer.append("g")
    let scrub_group = zoom_group.append('g')

    let plot = scrub_group.selectAll("polygon")
    .data(data.allocations_over_time)
    .enter()
    .append("polygon")
    .attr('points', format_points)
    .attr('fill', d => colors[d.color % colors.length])

    let axis = plot_coordinate_space.append('g').call(yaxis)


    let scale_mini = 0
    let translate_mini = 0
    function handleZoom(e) {
        const t = e.transform
        zoom_group.attr("transform", t)
        axis.call(yaxis.scale(e.transform.rescaleY(yscale)))
    }

    const thezoom = zoom().on('zoom', handleZoom)
    plot_outer.call(thezoom)

    return {
        select_window: (stepbegin, stepend, max) => {
            let begin = xscale(stepbegin)
            let size = xscale(stepend) - xscale(stepbegin);
            let scale = plot_width / size
            let translate = -begin
            let yscale =  max_size/max
            scrub_group.attr("transform", `scale(${scale/yscale}, 1) translate(${translate}, 0)`)
            plot_outer.call(thezoom.transform, zoomIdentity.scale(yscale).translate(0, -(plot_height - plot_height/yscale)))
        },
        set_delegate: (delegate) => {
            plot.on('mouseover', function (e, d) { delegate.set_selected(d3.select(this)) } )
            .on('mousedown', function(e, d) { delegate.default_selected = d3.select(this)})
            .on('mouseleave', function (e, d) { delegate.set_selected(delegate.default_selected) } )
        }
    }
}

function ContextViewer(text, data) {
    let current_selected = null

    return {
        default_selected: null,
        set_selected: (d) => {
            if (current_selected !== null) {
                current_selected.attr('stroke', null).attr('stroke-width', null);
            }
            if (d === null) {
                text.text("")
            } else {
                const dd = d.datum()
                if (dd.elem === 'summarized') {
                    text.html(
                        "Small tensors that were not plotted to cutdown on render time.\n" +
                        "Use detail slider to see smaller allocations.")
                } else {
                    text.text(`${dd.elem} ${data.context_for_id(dd.elem)}`)
                }
                d.attr('stroke', 'black').attr('stroke-width', 1).attr('vector-effect', 'non-scaling-stroke')
            }
            current_selected = d
        }
    }
}


function MiniMap(mini_svg, plot, data, left_pad, height=70) {
    let max_at_time = data.max_at_time
    let width = mini_svg.attr('width')
    let plot_width = width - left_pad
    let yscale = scaleLinear().domain([0, data.max_size]).range([height, 0]);
    let minixscale = scaleLinear().domain([0, max_at_time.length]).range([left_pad, width])

    let mini_points = [[max_at_time.length, 0], [0, 0]]

    for (const [i, m] of max_at_time.entries()) {
        let [lastx, lasty] = mini_points[mini_points.length - 1]
        if (m !== lasty) {
            mini_points.push([i, lasty])
            mini_points.push([i, m])
        } else if (i === max_at_time.length - 1) {
            mini_points.push([i, m])
        }
    }


    let points = mini_points.map(([t, o]) => `${minixscale(t)}, ${yscale(o)}`)
    points = points.join(' ')
    mini_svg.append('polygon').attr('points', points).attr('fill', schemeTableau10[0])

    let xscale = scaleLinear().domain([0, max_at_time.length]).range([0, plot_width])


    const brush = brushX()
    brush.extent([[left_pad, 0], [width, height]])
    brush.on('brush', function({selection}) {
        let [begin, end] = selection.map(x => x - left_pad)

        let stepbegin = Math.floor(xscale.invert(begin))
        let stepend = Math.floor(xscale.invert(end))
        let max = 0
        for (let i = stepbegin; i < stepend; i++) {
            max = Math.max(max, max_at_time[i])
        }
        plot.select_window(stepbegin, stepend, max)
    })
    mini_svg.call(brush)
    return {}
}

function Legend(plot_svg, categories, width) {
    let xstart = width - 100
    let ystart = 30
    plot_svg.append('g').selectAll('rect')
    .data(categories)
    .enter()
    .append('rect')
    .attr('x', (c, i) => xstart)
    .attr('y', (c, i) => ystart + i*15)
    .attr('width', 10)
    .attr('height', 10)
    .attr('fill', (c, i) => schemeTableau10[i % schemeTableau10.length])
    plot_svg.append('g').selectAll('text')
    .data(categories)
    .enter()
    .append('text')
    .attr('x', (c, i) => xstart + 20)
    .attr('y', (c, i) => ystart + i*15 + 8)
    .attr("font-family", "helvetica")
    .attr('font-size', 10)
    .text((c) => c)
    return {}
}


function create(max_entries) {
    let left_pad = 70
    let width = 1024
    let height = 768
    let data = process_alloc_data(max_entries)
    let body = d3.select("body")
    body.selectAll('svg').remove()
    body.selectAll('div').remove()

    if (alloc_data.elements_info.length > max_entries) {
         let d = body.append('div')
         d.append('input')
         .attr("type", "range")
         .attr('min', 0)
         .attr('max', alloc_data.elements_info.length)
         .attr("value", max_entries)
         .on('change', function() {
            create(this.value)
         })
         d.append('label').text('Detail')
    }

    let plot_svg = body.append("svg").attr('width', width).attr('height', height).attr('display', 'block')
    let plot = MemoryPlot(plot_svg, data, left_pad)

    if (alloc_data.categories !== null) {
        Legend(plot_svg.append('g'), alloc_data.categories, width)
    }

    MiniMap(body.append("svg").attr('width', width).attr('height', 80).attr('display', 'block'), plot, data, left_pad)
    let delegate = ContextViewer(body.append("div").append("pre").text('none'), data)
    plot.set_delegate(delegate)
}

export function main(data) {
    alloc_data = data
    create(15000)
}
