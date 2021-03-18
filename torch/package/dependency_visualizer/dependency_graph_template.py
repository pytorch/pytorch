TEMPLATE = """
<!DOCTYPE>

<html>

<head>
    <title>cytoscape-expand-collapse.js demo</title>

    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1, maximum-scale=1">

    <script src="https://unpkg.com/cytoscape/dist/cytoscape.min.js"></script>

    <!-- for testing with local version of cytoscape.js -->
    <script src="https://unpkg.com/layout-base/layout-base.js"></script>
    <script src="https://unpkg.com/cose-base/cose-base.js"></script>
    <script src="https://unpkg.com/cytoscape-cose-bilkent/cytoscape-cose-bilkent.js"></script>

    <style>
        body {
            font-family: helvetica neue, helvetica, liberation sans, arial, sans-serif;
            font-size: 14px;
        }

        #cy {
            z-index: 999;
            width: 85%;
            height: 85%;
            float: left;
        }

        h1 {
            opacity: 0.5;
            font-size: 1em;
            font-weight: bold;
        }
    </style>

    <script>
    (function (f) { if (typeof exports === "object" && typeof module !== "undefined") { module.exports = f() } else if (typeof define === "function" && define.amd) { define([], f) } else { var g; if (typeof window !== "undefined") { g = window } else if (typeof global !== "undefined") { g = global } else if (typeof self !== "undefined") { g = self } else { g = this } g.cytoscapeExpandCollapse = f() } })(function () {
        var define, module, exports; return (function e(t, n, r) { function s(o, u) { if (!n[o]) { if (!t[o]) { var a = typeof require == "function" && require; if (!u && a) return a(o, !0); if (i) return i(o, !0); var f = new Error("Cannot find module '" + o + "'"); throw f.code = "MODULE_NOT_FOUND", f } var l = n[o] = { exports: {} }; t[o][0].call(l.exports, function (e) { var n = t[o][1][e]; return s(n ? n : e) }, l, l.exports, e, t, n, r) } return n[o].exports } var i = typeof require == "function" && require; for (var o = 0; o < r.length; o++)s(r[o]); return s })({
            1: [function (_dereq_, module, exports) {
                var boundingBoxUtilities = {
                    equalBoundingBoxes: function (bb1, bb2) {
                        return bb1.x1 == bb2.x1 && bb1.x2 == bb2.x2 && bb1.y1 == bb2.y1 && bb1.y2 == bb2.y2;
                    },
                    getUnion: function (bb1, bb2) {
                        var union = {
                            x1: Math.min(bb1.x1, bb2.x1),
                            x2: Math.max(bb1.x2, bb2.x2),
                            y1: Math.min(bb1.y1, bb2.y1),
                            y2: Math.max(bb1.y2, bb2.y2),
                        };

                        union.w = union.x2 - union.x1;
                        union.h = union.y2 - union.y1;

                        return union;
                    }
                };

                module.exports = boundingBoxUtilities;
            }, {}], 2: [function (_dereq_, module, exports) {
                var debounce = _dereq_('./debounce');
                var debounce2 = _dereq_('./debounce2');

                module.exports = function (params, cy, api) {
                    var elementUtilities;
                    var fn = params;
                    const CUE_POS_UPDATE_DELAY = 100;
                    var nodeWithRenderedCue;

                    const getData = function () {
                        var scratch = cy.scratch('_cyExpandCollapse');
                        return scratch && scratch.cueUtilities;
                    };

                    const setData = function (data) {
                        var scratch = cy.scratch('_cyExpandCollapse');
                        if (scratch == null) {
                            scratch = {};
                        }

                        scratch.cueUtilities = data;
                        cy.scratch('_cyExpandCollapse', scratch);
                    };

                    var functions = {
                        init: function () {
                            var $canvas = document.createElement('canvas');
                            $canvas.classList.add("expand-collapse-canvas");
                            var $container = cy.container();
                            var ctx = $canvas.getContext('2d');
                            $container.append($canvas);

                            elementUtilities = _dereq_('./elementUtilities')(cy);

                            var offset = function (elt) {
                                var rect = elt.getBoundingClientRect();

                                return {
                                    top: rect.top + document.documentElement.scrollTop,
                                    left: rect.left + document.documentElement.scrollLeft
                                }
                            }

                            var _sizeCanvas = debounce(function () {
                                $canvas.height = cy.container().offsetHeight;
                                $canvas.width = cy.container().offsetWidth;
                                $canvas.style.position = 'absolute';
                                $canvas.style.top = 0;
                                $canvas.style.left = 0;
                                $canvas.style.zIndex = options().zIndex;

                                setTimeout(function () {
                                    var canvasBb = offset($canvas);
                                    var containerBb = offset($container);
                                    $canvas.style.top = -(canvasBb.top - containerBb.top);
                                    $canvas.style.left = -(canvasBb.left - containerBb.left);

                                    // refresh the cues on canvas resize
                                    if (cy) {
                                        clearDraws(true);
                                    }
                                }, 0);

                            }, 250);

                            function sizeCanvas() {
                                _sizeCanvas();
                            }

                            sizeCanvas();

                            var data = {};

                            // if there are events field in data unbind them here
                            // to prevent binding the same event multiple times
                            // if (!data.hasEventFields) {
                            //   functions['unbind'].apply( $container );
                            // }

                            function options() {
                                return cy.scratch('_cyExpandCollapse').options;
                            }

                            function clearDraws() {
                                var w = cy.width();
                                var h = cy.height();

                                ctx.clearRect(0, 0, w, h);
                                nodeWithRenderedCue = null;
                            }

                            function drawExpandCollapseCue(node) {
                                var children = node.children();
                                var collapsedChildren = node.data('collapsedChildren');
                                var hasChildren = children != null && children != undefined && children.length > 0;
                                // If this is a simple node with no collapsed children return directly
                                if (!hasChildren && !collapsedChildren) {
                                    return;
                                }

                                var isCollapsed = node.hasClass('cy-expand-collapse-collapsed-node');

                                //Draw expand-collapse rectangles
                                var rectSize = options().expandCollapseCueSize;
                                var lineSize = options().expandCollapseCueLineSize;

                                var cueCenter;

                                if (options().expandCollapseCuePosition === 'top-left') {
                                    var offset = 1;
                                    var size = cy.zoom() < 1 ? rectSize / (2 * cy.zoom()) : rectSize / 2;
                                    var nodeBorderWid = parseFloat(node.css('border-width'));
                                    var x = node.position('x') - node.width() / 2 - parseFloat(node.css('padding-left'))
                                        + nodeBorderWid + size + offset;
                                    var y = node.position('y') - node.height() / 2 - parseFloat(node.css('padding-top'))
                                        + nodeBorderWid + size + offset;

                                    cueCenter = { x: x, y: y };
                                } else {
                                    var option = options().expandCollapseCuePosition;
                                    cueCenter = typeof option === 'function' ? option.call(this, node) : option;
                                }

                                var expandcollapseCenter = elementUtilities.convertToRenderedPosition(cueCenter);

                                // convert to rendered sizes
                                rectSize = Math.max(rectSize, rectSize * cy.zoom());
                                lineSize = Math.max(lineSize, lineSize * cy.zoom());
                                var diff = (rectSize - lineSize) / 2;

                                var expandcollapseCenterX = expandcollapseCenter.x;
                                var expandcollapseCenterY = expandcollapseCenter.y;

                                var expandcollapseStartX = expandcollapseCenterX - rectSize / 2;
                                var expandcollapseStartY = expandcollapseCenterY - rectSize / 2;
                                var expandcollapseRectSize = rectSize;

                                // Draw expand/collapse cue if specified use an image else render it in the default way
                                if (isCollapsed && options().expandCueImage) {
                                    drawImg(options().expandCueImage, expandcollapseStartX, expandcollapseStartY, rectSize, rectSize);
                                }
                                else if (!isCollapsed && options().collapseCueImage) {
                                    drawImg(options().collapseCueImage, expandcollapseStartX, expandcollapseStartY, rectSize, rectSize);
                                }
                                else {
                                    var oldFillStyle = ctx.fillStyle;
                                    var oldWidth = ctx.lineWidth;
                                    var oldStrokeStyle = ctx.strokeStyle;

                                    ctx.fillStyle = "black";
                                    ctx.strokeStyle = "black";

                                    ctx.ellipse(expandcollapseCenterX, expandcollapseCenterY, rectSize / 2, rectSize / 2, 0, 0, 2 * Math.PI);
                                    ctx.fill();

                                    ctx.beginPath();

                                    ctx.strokeStyle = "white";
                                    ctx.lineWidth = Math.max(2.6, 2.6 * cy.zoom());

                                    ctx.moveTo(expandcollapseStartX + diff, expandcollapseStartY + rectSize / 2);
                                    ctx.lineTo(expandcollapseStartX + lineSize + diff, expandcollapseStartY + rectSize / 2);

                                    if (isCollapsed) {
                                        ctx.moveTo(expandcollapseStartX + rectSize / 2, expandcollapseStartY + diff);
                                        ctx.lineTo(expandcollapseStartX + rectSize / 2, expandcollapseStartY + lineSize + diff);
                                    }

                                    ctx.closePath();
                                    ctx.stroke();

                                    ctx.strokeStyle = oldStrokeStyle;
                                    ctx.fillStyle = oldFillStyle;
                                    ctx.lineWidth = oldWidth;
                                }

                                node._private.data.expandcollapseRenderedStartX = expandcollapseStartX;
                                node._private.data.expandcollapseRenderedStartY = expandcollapseStartY;
                                node._private.data.expandcollapseRenderedCueSize = expandcollapseRectSize;

                                nodeWithRenderedCue = node;
                            }

                            function drawImg(imgSrc, x, y, w, h) {
                                var img = new Image(w, h);
                                img.src = imgSrc;
                                img.onload = () => {
                                    ctx.drawImage(img, x, y, w, h);
                                };
                            }

                            cy.on('resize', data.eCyResize = function () {
                                sizeCanvas();
                            });

                            cy.on('expandcollapse.clearvisualcue', function () {
                                if (nodeWithRenderedCue) {
                                    clearDraws();
                                }
                            });

                            var oldMousePos = null, currMousePos = null;
                            cy.on('mousedown', data.eMouseDown = function (e) {
                                oldMousePos = e.renderedPosition || e.cyRenderedPosition
                            });

                            cy.on('mouseup', data.eMouseUp = function (e) {
                                currMousePos = e.renderedPosition || e.cyRenderedPosition
                            });

                            cy.on('remove', 'node', data.eRemove = function () {
                                clearDraws();
                            });

                            var ur;
                            cy.on('select unselect', data.eSelect = function () {
                                if (nodeWithRenderedCue) {
                                    clearDraws();
                                }
                                var isOnly1Selected = cy.$(':selected').length == 1;
                                var isOnly1SelectedCompundNode = cy.nodes(':parent').filter(':selected').length == 1 && isOnly1Selected;
                                var isOnly1SelectedCollapsedNode = cy.nodes('.cy-expand-collapse-collapsed-node').filter(':selected').length == 1 && isOnly1Selected;
                                if (isOnly1SelectedCollapsedNode || isOnly1SelectedCompundNode) {
                                    drawExpandCollapseCue(cy.nodes(':selected')[0]);
                                }
                            });

                            cy.on('tap', data.eTap = function (event) {
                                var node = nodeWithRenderedCue;
                                if (!node) {
                                    return;
                                }
                                var expandcollapseRenderedStartX = node.data('expandcollapseRenderedStartX');
                                var expandcollapseRenderedStartY = node.data('expandcollapseRenderedStartY');
                                var expandcollapseRenderedRectSize = node.data('expandcollapseRenderedCueSize');
                                var expandcollapseRenderedEndX = expandcollapseRenderedStartX + expandcollapseRenderedRectSize;
                                var expandcollapseRenderedEndY = expandcollapseRenderedStartY + expandcollapseRenderedRectSize;

                                var cyRenderedPos = event.renderedPosition || event.cyRenderedPosition;
                                var cyRenderedPosX = cyRenderedPos.x;
                                var cyRenderedPosY = cyRenderedPos.y;
                                var opts = options();
                                var factor = (opts.expandCollapseCueSensitivity - 1) / 2;

                                if ((Math.abs(oldMousePos.x - currMousePos.x) < 5 && Math.abs(oldMousePos.y - currMousePos.y) < 5)
                                    && cyRenderedPosX >= expandcollapseRenderedStartX - expandcollapseRenderedRectSize * factor
                                    && cyRenderedPosX <= expandcollapseRenderedEndX + expandcollapseRenderedRectSize * factor
                                    && cyRenderedPosY >= expandcollapseRenderedStartY - expandcollapseRenderedRectSize * factor
                                    && cyRenderedPosY <= expandcollapseRenderedEndY + expandcollapseRenderedRectSize * factor) {
                                    if (opts.undoable && !ur) {
                                        ur = cy.undoRedo({ defaultActions: false });
                                    }

                                    if (api.isCollapsible(node)) {
                                        clearDraws();
                                        if (opts.undoable) {
                                            ur.do("collapse", {
                                                nodes: node,
                                                options: opts
                                            });
                                        }
                                        else {
                                            api.collapse(node, opts);
                                        }
                                    }
                                    else if (api.isExpandable(node)) {
                                        clearDraws();
                                        if (opts.undoable) {
                                            ur.do("expand", { nodes: node, options: opts });
                                        }
                                        else {
                                            api.expand(node, opts);
                                        }
                                    }
                                    if (node.selectable()) {
                                        node.unselectify();
                                        cy.scratch('_cyExpandCollapse').selectableChanged = true;
                                    }
                                }
                            });

                            cy.on('position', 'node', data.ePosition = debounce2(data.eSelect, CUE_POS_UPDATE_DELAY, clearDraws));

                            cy.on('pan zoom', data.ePosition);

                            cy.on('expandcollapse.afterexpand expandcollapse.aftercollapse', 'node', data.eAfterExpandCollapse = function () {
                                var delay = 50 + params.animate ? params.animationDuration : 0;
                                setTimeout(() => {
                                    if (this.selected()) {
                                        drawExpandCollapseCue(this);
                                    }
                                }, delay);
                            });

                            // write options to data
                            data.hasEventFields = true;
                            setData(data);
                        },
                        unbind: function () {
                            // var $container = this;
                            var data = getData();

                            if (!data.hasEventFields) {
                                console.log('events to unbind does not exist');
                                return;
                            }

                            cy.trigger('expandcollapse.clearvisualcue');

                            cy.off('mousedown', 'node', data.eMouseDown)
                                .off('mouseup', 'node', data.eMouseUp)
                                .off('remove', 'node', data.eRemove)
                                .off('tap', 'node', data.eTap)
                                .off('add', 'node', data.eAdd)
                                .off('position', 'node', data.ePosition)
                                .off('pan zoom', data.ePosition)
                                .off('select unselect', data.eSelect)
                                .off('expandcollapse.afterexpand expandcollapse.aftercollapse', 'node', data.eAfterExpandCollapse)
                                .off('free', 'node', data.eFree)
                                .off('resize', data.eCyResize);
                        },
                        rebind: function () {
                            var data = getData();

                            if (!data.hasEventFields) {
                                console.log('events to rebind does not exist');
                                return;
                            }

                            cy.on('mousedown', 'node', data.eMouseDown)
                                .on('mouseup', 'node', data.eMouseUp)
                                .on('remove', 'node', data.eRemove)
                                .on('tap', 'node', data.eTap)
                                .on('add', 'node', data.eAdd)
                                .on('position', 'node', data.ePosition)
                                .on('pan zoom', data.ePosition)
                                .on('select unselect', data.eSelect)
                                .on('expandcollapse.afterexpand expandcollapse.aftercollapse', 'node', data.eAfterExpandCollapse)
                                .on('free', 'node', data.eFree)
                                .on('resize', data.eCyResize);
                        }
                    };

                    if (functions[fn]) {
                        return functions[fn].apply(cy.container(), Array.prototype.slice.call(arguments, 1));
                    } else if (typeof fn == 'object' || !fn) {
                        return functions.init.apply(cy.container(), arguments);
                    }
                    throw new Error('No such function `' + fn + '` for cytoscape.js-expand-collapse');

                };

            }, { "./debounce": 3, "./debounce2": 4, "./elementUtilities": 5 }], 3: [function (_dereq_, module, exports) {
                var debounce = (function () {
                    /**
                     * lodash 3.1.1 (Custom Build) <https://lodash.com/>
                     * Build: `lodash modern modularize exports="npm" -o ./`
                     * Copyright 2012-2015 The Dojo Foundation <http://dojofoundation.org/>
                     * Based on Underscore.js 1.8.3 <http://underscorejs.org/LICENSE>
                     * Copyright 2009-2015 Jeremy Ashkenas, DocumentCloud and Investigative Reporters & Editors
                     * Available under MIT license <https://lodash.com/license>
                     */
                    /** Used as the `TypeError` message for "Functions" methods. */
                    var FUNC_ERROR_TEXT = 'Expected a function';

                    /* Native method references for those with the same name as other `lodash` methods. */
                    var nativeMax = Math.max,
                        nativeNow = Date.now;

                    /**
                     * Gets the number of milliseconds that have elapsed since the Unix epoch
                     * (1 January 1970 00:00:00 UTC).
                     *
                     * @static
                     * @memberOf _
                     * @category Date
                     * @example
                     *
                     * _.defer(function(stamp) {
                     *   console.log(_.now() - stamp);
                     * }, _.now());
                     * // => logs the number of milliseconds it took for the deferred function to be invoked
                     */
                    var now = nativeNow || function () {
                        return new Date().getTime();
                    };

                    /**
                     * Creates a debounced function that delays invoking `func` until after `wait`
                     * milliseconds have elapsed since the last time the debounced function was
                     * invoked. The debounced function comes with a `cancel` method to cancel
                     * delayed invocations. Provide an options object to indicate that `func`
                     * should be invoked on the leading and/or trailing edge of the `wait` timeout.
                     * Subsequent calls to the debounced function return the result of the last
                     * `func` invocation.
                     *
                     * **Note:** If `leading` and `trailing` options are `true`, `func` is invoked
                     * on the trailing edge of the timeout only if the the debounced function is
                     * invoked more than once during the `wait` timeout.
                     *
                     * See [David Corbacho's article](http://drupalmotion.com/article/debounce-and-throttle-visual-explanation)
                     * for details over the differences between `_.debounce` and `_.throttle`.
                     *
                     * @static
                     * @memberOf _
                     * @category Function
                     * @param {Function} func The function to debounce.
                     * @param {number} [wait=0] The number of milliseconds to delay.
                     * @param {Object} [options] The options object.
                     * @param {boolean} [options.leading=false] Specify invoking on the leading
                     *  edge of the timeout.
                     * @param {number} [options.maxWait] The maximum time `func` is allowed to be
                     *  delayed before it's invoked.
                     * @param {boolean} [options.trailing=true] Specify invoking on the trailing
                     *  edge of the timeout.
                     * @returns {Function} Returns the new debounced function.
                     * @example
                     *
                     * // avoid costly calculations while the window size is in flux
                     * jQuery(window).on('resize', _.debounce(calculateLayout, 150));
                     *
                     * // invoke `sendMail` when the click event is fired, debouncing subsequent calls
                     * jQuery('#postbox').on('click', _.debounce(sendMail, 300, {
                     *   'leading': true,
                     *   'trailing': false
                     * }));
                     *
                     * // ensure `batchLog` is invoked once after 1 second of debounced calls
                     * var source = new EventSource('/stream');
                     * jQuery(source).on('message', _.debounce(batchLog, 250, {
                     *   'maxWait': 1000
                     * }));
                     *
                     * // cancel a debounced call
                     * var todoChanges = _.debounce(batchLog, 1000);
                     * Object.observe(models.todo, todoChanges);
                     *
                     * Object.observe(models, function(changes) {
                     *   if (_.find(changes, { 'user': 'todo', 'type': 'delete'})) {
                     *     todoChanges.cancel();
                     *   }
                     * }, ['delete']);
                     *
                     * // ...at some point `models.todo` is changed
                     * models.todo.completed = true;
                     *
                     * // ...before 1 second has passed `models.todo` is deleted
                     * // which cancels the debounced `todoChanges` call
                     * delete models.todo;
                     */
                    function debounce(func, wait, options) {
                        var args,
                            maxTimeoutId,
                            result,
                            stamp,
                            thisArg,
                            timeoutId,
                            trailingCall,
                            lastCalled = 0,
                            maxWait = false,
                            trailing = true;

                        if (typeof func != 'function') {
                            throw new TypeError(FUNC_ERROR_TEXT);
                        }
                        wait = wait < 0 ? 0 : (+wait || 0);
                        if (options === true) {
                            var leading = true;
                            trailing = false;
                        } else if (isObject(options)) {
                            leading = !!options.leading;
                            maxWait = 'maxWait' in options && nativeMax(+options.maxWait || 0, wait);
                            trailing = 'trailing' in options ? !!options.trailing : trailing;
                        }

                        function cancel() {
                            if (timeoutId) {
                                clearTimeout(timeoutId);
                            }
                            if (maxTimeoutId) {
                                clearTimeout(maxTimeoutId);
                            }
                            lastCalled = 0;
                            maxTimeoutId = timeoutId = trailingCall = undefined;
                        }

                        function complete(isCalled, id) {
                            if (id) {
                                clearTimeout(id);
                            }
                            maxTimeoutId = timeoutId = trailingCall = undefined;
                            if (isCalled) {
                                lastCalled = now();
                                result = func.apply(thisArg, args);
                                if (!timeoutId && !maxTimeoutId) {
                                    args = thisArg = undefined;
                                }
                            }
                        }

                        function delayed() {
                            var remaining = wait - (now() - stamp);
                            if (remaining <= 0 || remaining > wait) {
                                complete(trailingCall, maxTimeoutId);
                            } else {
                                timeoutId = setTimeout(delayed, remaining);
                            }
                        }

                        function maxDelayed() {
                            complete(trailing, timeoutId);
                        }

                        function debounced() {
                            args = arguments;
                            stamp = now();
                            thisArg = this;
                            trailingCall = trailing && (timeoutId || !leading);

                            if (maxWait === false) {
                                var leadingCall = leading && !timeoutId;
                            } else {
                                if (!maxTimeoutId && !leading) {
                                    lastCalled = stamp;
                                }
                                var remaining = maxWait - (stamp - lastCalled),
                                    isCalled = remaining <= 0 || remaining > maxWait;

                                if (isCalled) {
                                    if (maxTimeoutId) {
                                        maxTimeoutId = clearTimeout(maxTimeoutId);
                                    }
                                    lastCalled = stamp;
                                    result = func.apply(thisArg, args);
                                }
                                else if (!maxTimeoutId) {
                                    maxTimeoutId = setTimeout(maxDelayed, remaining);
                                }
                            }
                            if (isCalled && timeoutId) {
                                timeoutId = clearTimeout(timeoutId);
                            }
                            else if (!timeoutId && wait !== maxWait) {
                                timeoutId = setTimeout(delayed, wait);
                            }
                            if (leadingCall) {
                                isCalled = true;
                                result = func.apply(thisArg, args);
                            }
                            if (isCalled && !timeoutId && !maxTimeoutId) {
                                args = thisArg = undefined;
                            }
                            return result;
                        }

                        debounced.cancel = cancel;
                        return debounced;
                    }

                    /**
                     * Checks if `value` is the [language type](https://es5.github.io/#x8) of `Object`.
                     * (e.g. arrays, functions, objects, regexes, `new Number(0)`, and `new String('')`)
                     *
                     * @static
                     * @memberOf _
                     * @category Lang
                     * @param {*} value The value to check.
                     * @returns {boolean} Returns `true` if `value` is an object, else `false`.
                     * @example
                     *
                     * _.isObject({});
                     * // => true
                     *
                     * _.isObject([1, 2, 3]);
                     * // => true
                     *
                     * _.isObject(1);
                     * // => false
                     */
                    function isObject(value) {
                        // Avoid a V8 JIT bug in Chrome 19-20.
                        // See https://code.google.com/p/v8/issues/detail?id=2291 for more details.
                        var type = typeof value;
                        return !!value && (type == 'object' || type == 'function');
                    }

                    return debounce;

                })();

                module.exports = debounce;
            }, {}], 4: [function (_dereq_, module, exports) {
                var debounce2 = (function () {
                    /**
                     * Slightly modified version of debounce. Calls fn2 at the beginning of frequent calls to fn1
                     * @static
                     * @category Function
                     * @param {Function} fn1 The function to debounce.
                     * @param {number} [wait=0] The number of milliseconds to delay.
                     * @param {Function} fn2 The function to call the beginning of frequent calls to fn1
                     * @returns {Function} Returns the new debounced function.
                     */
                    function debounce2(fn1, wait, fn2) {
                        let timeout;
                        let isInit = true;
                        return function () {
                            const context = this, args = arguments;
                            const later = function () {
                                timeout = null;
                                fn1.apply(context, args);
                                isInit = true;
                            };
                            clearTimeout(timeout);
                            timeout = setTimeout(later, wait);
                            if (isInit) {
                                fn2.apply(context, args);
                                isInit = false;
                            }
                        };
                    }
                    return debounce2;
                })();

                module.exports = debounce2;
            }, {}], 5: [function (_dereq_, module, exports) {
                function elementUtilities(cy) {
                    return {
                        moveNodes: function (positionDiff, nodes, notCalcTopMostNodes) {
                            var topMostNodes = notCalcTopMostNodes ? nodes : this.getTopMostNodes(nodes);
                            var nonParents = topMostNodes.not(":parent");
                            // moving parents spoils positioning, so move only nonparents
                            nonParents.positions(function (ele, i) {
                                return {
                                    x: nonParents[i].position("x") + positionDiff.x,
                                    y: nonParents[i].position("y") + positionDiff.y
                                };
                            });
                            for (var i = 0; i < topMostNodes.length; i++) {
                                var node = topMostNodes[i];
                                var children = node.children();
                                this.moveNodes(positionDiff, children, true);
                            }
                        },
                        getTopMostNodes: function (nodes) {//*//
                            var nodesMap = {};
                            for (var i = 0; i < nodes.length; i++) {
                                nodesMap[nodes[i].id()] = true;
                            }
                            var roots = nodes.filter(function (ele, i) {
                                if (typeof ele === "number") {
                                    ele = i;
                                }

                                var parent = ele.parent()[0];
                                while (parent != null) {
                                    if (nodesMap[parent.id()]) {
                                        return false;
                                    }
                                    parent = parent.parent()[0];
                                }
                                return true;
                            });

                            return roots;
                        },
                        rearrange: function (layoutBy) {
                            if (typeof layoutBy === "function") {
                                layoutBy();
                            } else if (layoutBy != null) {
                                var layout = cy.layout(layoutBy);
                                if (layout && layout.run) {
                                    layout.run();
                                }
                            }
                        },
                        convertToRenderedPosition: function (modelPosition) {
                            var pan = cy.pan();
                            var zoom = cy.zoom();

                            var x = modelPosition.x * zoom + pan.x;
                            var y = modelPosition.y * zoom + pan.y;

                            return {
                                x: x,
                                y: y
                            };
                        }
                    };
                }

                module.exports = elementUtilities;

            }, {}], 6: [function (_dereq_, module, exports) {
                var boundingBoxUtilities = _dereq_('./boundingBoxUtilities');

                // Expand collapse utilities
                function expandCollapseUtilities(cy) {
                    var elementUtilities = _dereq_('./elementUtilities')(cy);
                    return {
                        //the number of nodes moving animatedly after expand operation
                        animatedlyMovingNodeCount: 0,
                        /*
                         * A funtion basicly expanding a node, it is to be called when a node is expanded anyway.
                         * Single parameter indicates if the node is expanded alone and if it is truthy then layoutBy parameter is considered to
                         * perform layout after expand.
                         */
                        expandNodeBaseFunction: function (node, single, layoutBy) {
                            if (!node._private.data.collapsedChildren) {
                                return;
                            }

                            //check how the position of the node is changed
                            var positionDiff = {
                                x: node._private.position.x - node._private.data['position-before-collapse'].x,
                                y: node._private.position.y - node._private.data['position-before-collapse'].y
                            };

                            node.removeData("infoLabel");
                            node.removeClass('cy-expand-collapse-collapsed-node');

                            node.trigger("expandcollapse.beforeexpand");
                            var restoredNodes = node._private.data.collapsedChildren;
                            restoredNodes.restore();
                            var parentData = cy.scratch('_cyExpandCollapse').parentData;
                            for (var i = 0; i < restoredNodes.length; i++) {
                                delete parentData[restoredNodes[i].id()];
                            }
                            cy.scratch('_cyExpandCollapse').parentData = parentData;
                            this.repairEdges(node);
                            node._private.data.collapsedChildren = null;

                            elementUtilities.moveNodes(positionDiff, node.children());
                            node.removeData('position-before-collapse');

                            node.trigger("position"); // position not triggered by default when nodes are moved
                            node.trigger("expandcollapse.afterexpand");

                            // If expand is called just for one node then call end operation to perform layout
                            if (single) {
                                this.endOperation(layoutBy, node);
                            }
                        },
                        /*
                         * A helper function to collapse given nodes in a simple way (Without performing layout afterward)
                         * It collapses all root nodes bottom up.
                         */
                        simpleCollapseGivenNodes: function (nodes) {//*//
                            nodes.data("collapse", true);
                            var roots = elementUtilities.getTopMostNodes(nodes);
                            for (var i = 0; i < roots.length; i++) {
                                var root = roots[i];

                                // Collapse the nodes in bottom up order
                                this.collapseBottomUp(root);
                            }

                            return nodes;
                        },
                        /*
                         * A helper function to expand given nodes in a simple way (Without performing layout afterward)
                         * It expands all top most nodes top down.
                         */
                        simpleExpandGivenNodes: function (nodes, applyFishEyeViewToEachNode) {
                            nodes.data("expand", true); // Mark that the nodes are still to be expanded
                            var roots = elementUtilities.getTopMostNodes(nodes);
                            for (var i = 0; i < roots.length; i++) {
                                var root = roots[i];
                                this.expandTopDown(root, applyFishEyeViewToEachNode); // For each root node expand top down
                            }
                            return nodes;
                        },
                        /*
                         * Expands all nodes by expanding all top most nodes top down with their descendants.
                         */
                        simpleExpandAllNodes: function (nodes, applyFishEyeViewToEachNode) {
                            if (nodes === undefined) {
                                nodes = cy.nodes();
                            }
                            var orphans;
                            orphans = elementUtilities.getTopMostNodes(nodes);
                            var expandStack = [];
                            for (var i = 0; i < orphans.length; i++) {
                                var root = orphans[i];
                                this.expandAllTopDown(root, expandStack, applyFishEyeViewToEachNode);
                            }
                            return expandStack;
                        },
                        /*
                         * The operation to be performed after expand/collapse. It rearrange nodes by layoutBy parameter.
                         */
                        endOperation: function (layoutBy, nodes) {
                            var self = this;
                            cy.ready(function () {
                                setTimeout(function () {
                                    elementUtilities.rearrange(layoutBy);
                                    if (cy.scratch('_cyExpandCollapse').selectableChanged) {
                                        nodes.selectify();
                                        cy.scratch('_cyExpandCollapse').selectableChanged = false;
                                    }
                                }, 0);

                            });
                        },
                        /*
                         * Calls simple expandAllNodes. Then performs end operation.
                         */
                        expandAllNodes: function (nodes, options) {//*//
                            var expandedStack = this.simpleExpandAllNodes(nodes, options.fisheye);

                            this.endOperation(options.layoutBy, nodes);

                            /*
                             * return the nodes to undo the operation
                             */
                            return expandedStack;
                        },
                        /*
                         * Expands the root and its collapsed descendents in top down order.
                         */
                        expandAllTopDown: function (root, expandStack, applyFishEyeViewToEachNode) {
                            if (root._private.data.collapsedChildren != null) {
                                expandStack.push(root);
                                this.expandNode(root, applyFishEyeViewToEachNode);
                            }
                            var children = root.children();
                            for (var i = 0; i < children.length; i++) {
                                var node = children[i];
                                this.expandAllTopDown(node, expandStack, applyFishEyeViewToEachNode);
                            }
                        },
                        //Expand the given nodes perform end operation after expandation
                        expandGivenNodes: function (nodes, options) {
                            // If there is just one node to expand we need to animate for fisheye view, but if there are more then one node we do not
                            if (nodes.length === 1) {

                                var node = nodes[0];
                                if (node._private.data.collapsedChildren != null) {
                                    // Expand the given node the third parameter indicates that the node is simple which ensures that fisheye parameter will be considered
                                    this.expandNode(node, options.fisheye, true, options.animate, options.layoutBy, options.animationDuration);
                                }
                            }
                            else {
                                // First expand given nodes and then perform layout according to the layoutBy parameter
                                this.simpleExpandGivenNodes(nodes, options.fisheye);
                                this.endOperation(options.layoutBy, nodes);
                            }

                            /*
                             * return the nodes to undo the operation
                             */
                            return nodes;
                        },
                        //collapse the given nodes then perform end operation
                        collapseGivenNodes: function (nodes, options) {
                            /*
                             * In collapse operation there is no fisheye view to be applied so there is no animation to be destroyed here. We can do this
                             * in a batch.
                             */
                            cy.startBatch();
                            this.simpleCollapseGivenNodes(nodes/*, options*/);
                            cy.endBatch();

                            nodes.trigger("position"); // position not triggered by default when collapseNode is called
                            this.endOperation(options.layoutBy, nodes);

                            // Update the style
                            cy.style().update();

                            /*
                             * return the nodes to undo the operation
                             */
                            return nodes;
                        },
                        //collapse the nodes in bottom up order starting from the root
                        collapseBottomUp: function (root) {
                            var children = root.children();
                            for (var i = 0; i < children.length; i++) {
                                var node = children[i];
                                this.collapseBottomUp(node);
                            }
                            //If the root is a compound node to be collapsed then collapse it
                            if (root.data("collapse") && root.children().length > 0) {
                                this.collapseNode(root);
                                root.removeData("collapse");
                            }
                        },
                        //expand the nodes in top down order starting from the root
                        expandTopDown: function (root, applyFishEyeViewToEachNode) {
                            if (root.data("expand") && root._private.data.collapsedChildren != null) {
                                // Expand the root and unmark its expand data to specify that it is no more to be expanded
                                this.expandNode(root, applyFishEyeViewToEachNode);
                                root.removeData("expand");
                            }
                            // Make a recursive call for children of root
                            var children = root.children();
                            for (var i = 0; i < children.length; i++) {
                                var node = children[i];
                                this.expandTopDown(node);
                            }
                        },
                        // Converst the rendered position to model position according to global pan and zoom values
                        convertToModelPosition: function (renderedPosition) {
                            var pan = cy.pan();
                            var zoom = cy.zoom();

                            var x = (renderedPosition.x - pan.x) / zoom;
                            var y = (renderedPosition.y - pan.y) / zoom;

                            return {
                                x: x,
                                y: y
                            };
                        },
                        /*
                         * This method expands the given node. It considers applyFishEyeView, animate and layoutBy parameters.
                         * It also considers single parameter which indicates if this node is expanded alone. If this parameter is truthy along with
                         * applyFishEyeView parameter then the state of view port is to be changed to have extra space on the screen (if needed) before appliying the
                         * fisheye view.
                         */
                        expandNode: function (node, applyFishEyeView, single, animate, layoutBy, animationDuration) {
                            var self = this;

                            var commonExpandOperation = function (node, applyFishEyeView, single, animate, layoutBy, animationDuration) {
                                if (applyFishEyeView) {

                                    node._private.data['width-before-fisheye'] = node._private.data['size-before-collapse'].w;
                                    node._private.data['height-before-fisheye'] = node._private.data['size-before-collapse'].h;

                                    // Fisheye view expand the node.
                                    // The first paramter indicates the node to apply fisheye view, the third parameter indicates the node
                                    // to be expanded after fisheye view is applied.
                                    self.fishEyeViewExpandGivenNode(node, single, node, animate, layoutBy, animationDuration);
                                }

                                // If one of these parameters is truthy it means that expandNodeBaseFunction is already to be called.
                                // However if none of them is truthy we need to call it here.
                                if (!single || !applyFishEyeView || !animate) {
                                    self.expandNodeBaseFunction(node, single, layoutBy);
                                }
                            };

                            if (node._private.data.collapsedChildren != null) {
                                this.storeWidthHeight(node);
                                var animating = false; // Variable to check if there is a current animation, if there is commonExpandOperation will be called after animation

                                // If the node is the only node to expand and fisheye view should be applied, then change the state of viewport
                                // to create more space on screen (If needed)
                                if (applyFishEyeView && single) {
                                    var topLeftPosition = this.convertToModelPosition({ x: 0, y: 0 });
                                    var bottomRightPosition = this.convertToModelPosition({ x: cy.width(), y: cy.height() });
                                    var padding = 80;
                                    var bb = {
                                        x1: topLeftPosition.x,
                                        x2: bottomRightPosition.x,
                                        y1: topLeftPosition.y,
                                        y2: bottomRightPosition.y
                                    };

                                    var nodeBB = {
                                        x1: node._private.position.x - node._private.data['size-before-collapse'].w / 2 - padding,
                                        x2: node._private.position.x + node._private.data['size-before-collapse'].w / 2 + padding,
                                        y1: node._private.position.y - node._private.data['size-before-collapse'].h / 2 - padding,
                                        y2: node._private.position.y + node._private.data['size-before-collapse'].h / 2 + padding
                                    };

                                    var unionBB = boundingBoxUtilities.getUnion(nodeBB, bb);

                                    // If these bboxes are not equal then we need to change the viewport state (by pan and zoom)
                                    if (!boundingBoxUtilities.equalBoundingBoxes(unionBB, bb)) {
                                        var viewPort = cy.getFitViewport(unionBB, 10);
                                        var self = this;
                                        animating = animate; // Signal that there is an animation now and commonExpandOperation will be called after animation
                                        // Check if we need to animate during pan and zoom
                                        if (animate) {
                                            cy.animate({
                                                pan: viewPort.pan,
                                                zoom: viewPort.zoom,
                                                complete: function () {
                                                    commonExpandOperation(node, applyFishEyeView, single, animate, layoutBy, animationDuration);
                                                }
                                            }, {
                                                duration: animationDuration || 1000
                                            });
                                        }
                                        else {
                                            cy.zoom(viewPort.zoom);
                                            cy.pan(viewPort.pan);
                                        }
                                    }
                                }

                                // If animating is not true we need to call commonExpandOperation here
                                if (!animating) {
                                    commonExpandOperation(node, applyFishEyeView, single, animate, layoutBy, animationDuration);
                                }

                                //return the node to undo the operation
                                return node;
                            }
                        },
                        //collapse the given node without performing end operation
                        collapseNode: function (node) {
                            if (node._private.data.collapsedChildren == null) {
                                node.data('position-before-collapse', {
                                    x: node.position().x,
                                    y: node.position().y
                                });

                                node.data('size-before-collapse', {
                                    w: node.outerWidth(),
                                    h: node.outerHeight()
                                });

                                var children = node.children();

                                children.unselect();
                                children.connectedEdges().unselect();

                                node.trigger("expandcollapse.beforecollapse");

                                this.barrowEdgesOfcollapsedChildren(node);
                                this.removeChildren(node, node);
                                node.addClass('cy-expand-collapse-collapsed-node');

                                node.trigger("expandcollapse.aftercollapse");

                                node.position(node.data('position-before-collapse'));

                                //return the node to undo the operation
                                return node;
                            }
                        },
                        storeWidthHeight: function (node) {//*//
                            if (node != null) {
                                node._private.data['x-before-fisheye'] = this.xPositionInParent(node);
                                node._private.data['y-before-fisheye'] = this.yPositionInParent(node);
                                node._private.data['width-before-fisheye'] = node.outerWidth();
                                node._private.data['height-before-fisheye'] = node.outerHeight();

                                if (node.parent()[0] != null) {
                                    this.storeWidthHeight(node.parent()[0]);
                                }
                            }

                        },
                        /*
                         * Apply fisheye view to the given node. nodeToExpand will be expanded after the operation.
                         * The other parameter are to be passed by parameters directly in internal function calls.
                         */
                        fishEyeViewExpandGivenNode: function (node, single, nodeToExpand, animate, layoutBy, animationDuration) {
                            var siblings = this.getSiblings(node);

                            var x_a = this.xPositionInParent(node);
                            var y_a = this.yPositionInParent(node);

                            var d_x_left = Math.abs((node._private.data['width-before-fisheye'] - node.outerWidth()) / 2);
                            var d_x_right = Math.abs((node._private.data['width-before-fisheye'] - node.outerWidth()) / 2);
                            var d_y_upper = Math.abs((node._private.data['height-before-fisheye'] - node.outerHeight()) / 2);
                            var d_y_lower = Math.abs((node._private.data['height-before-fisheye'] - node.outerHeight()) / 2);

                            var abs_diff_on_x = Math.abs(node._private.data['x-before-fisheye'] - x_a);
                            var abs_diff_on_y = Math.abs(node._private.data['y-before-fisheye'] - y_a);

                            // Center went to LEFT
                            if (node._private.data['x-before-fisheye'] > x_a) {
                                d_x_left = d_x_left + abs_diff_on_x;
                                d_x_right = d_x_right - abs_diff_on_x;
                            }
                            // Center went to RIGHT
                            else {
                                d_x_left = d_x_left - abs_diff_on_x;
                                d_x_right = d_x_right + abs_diff_on_x;
                            }

                            // Center went to UP
                            if (node._private.data['y-before-fisheye'] > y_a) {
                                d_y_upper = d_y_upper + abs_diff_on_y;
                                d_y_lower = d_y_lower - abs_diff_on_y;
                            }
                            // Center went to DOWN
                            else {
                                d_y_upper = d_y_upper - abs_diff_on_y;
                                d_y_lower = d_y_lower + abs_diff_on_y;
                            }

                            var xPosInParentSibling = [];
                            var yPosInParentSibling = [];

                            for (var i = 0; i < siblings.length; i++) {
                                xPosInParentSibling.push(this.xPositionInParent(siblings[i]));
                                yPosInParentSibling.push(this.yPositionInParent(siblings[i]));
                            }

                            for (var i = 0; i < siblings.length; i++) {
                                var sibling = siblings[i];

                                var x_b = xPosInParentSibling[i];
                                var y_b = yPosInParentSibling[i];

                                var slope = (y_b - y_a) / (x_b - x_a);

                                var d_x = 0;
                                var d_y = 0;
                                var T_x = 0;
                                var T_y = 0;

                                // Current sibling is on the LEFT
                                if (x_a > x_b) {
                                    d_x = d_x_left;
                                }
                                // Current sibling is on the RIGHT
                                else {
                                    d_x = d_x_right;
                                }
                                // Current sibling is on the UPPER side
                                if (y_a > y_b) {
                                    d_y = d_y_upper;
                                }
                                // Current sibling is on the LOWER side
                                else {
                                    d_y = d_y_lower;
                                }

                                if (isFinite(slope)) {
                                    T_x = Math.min(d_x, (d_y / Math.abs(slope)));
                                }

                                if (slope !== 0) {
                                    T_y = Math.min(d_y, (d_x * Math.abs(slope)));
                                }

                                if (x_a > x_b) {
                                    T_x = -1 * T_x;
                                }

                                if (y_a > y_b) {
                                    T_y = -1 * T_y;
                                }

                                // Move the sibling in the special way
                                this.fishEyeViewMoveNode(sibling, T_x, T_y, nodeToExpand, single, animate, layoutBy, animationDuration);
                            }

                            // If there is no sibling call expand node base function here else it is to be called one of fishEyeViewMoveNode() calls
                            if (siblings.length == 0) {
                                this.expandNodeBaseFunction(nodeToExpand, single, layoutBy);
                            }

                            if (node.parent()[0] != null) {
                                // Apply fisheye view to the parent node as well ( If exists )
                                this.fishEyeViewExpandGivenNode(node.parent()[0], single, nodeToExpand, animate, layoutBy, animationDuration);
                            }

                            return node;
                        },
                        getSiblings: function (node) {
                            var siblings;

                            if (node.parent()[0] == null) {
                                var orphans = cy.nodes(":visible").orphans();
                                siblings = orphans.difference(node);
                            } else {
                                siblings = node.siblings(":visible");
                            }

                            return siblings;
                        },
                        /*
                         * Move node operation specialized for fish eye view expand operation
                         * Moves the node by moving its descandents. Movement is animated if both single and animate flags are truthy.
                         */
                        fishEyeViewMoveNode: function (node, T_x, T_y, nodeToExpand, single, animate, layoutBy, animationDuration) {
                            var childrenList = cy.collection();
                            if (node.isParent()) {
                                childrenList = node.children(":visible");
                            }
                            var self = this;

                            /*
                             * If the node is simple move itself directly else move it by moving its children by a self recursive call
                             */
                            if (childrenList.length == 0) {
                                var newPosition = { x: node._private.position.x + T_x, y: node._private.position.y + T_y };
                                if (!single || !animate) {
                                    node._private.position.x = newPosition.x;
                                    node._private.position.y = newPosition.y;
                                }
                                else {
                                    this.animatedlyMovingNodeCount++;
                                    node.animate({
                                        position: newPosition,
                                        complete: function () {
                                            self.animatedlyMovingNodeCount--;
                                            if (self.animatedlyMovingNodeCount > 0 || !nodeToExpand.hasClass('cy-expand-collapse-collapsed-node')) {

                                                return;
                                            }

                                            // If all nodes are moved we are ready to expand so call expand node base function
                                            self.expandNodeBaseFunction(nodeToExpand, single, layoutBy);

                                        }
                                    }, {
                                        duration: animationDuration || 1000
                                    });
                                }
                            }
                            else {
                                for (var i = 0; i < childrenList.length; i++) {
                                    this.fishEyeViewMoveNode(childrenList[i], T_x, T_y, nodeToExpand, single, animate, layoutBy, animationDuration);
                                }
                            }
                        },
                        xPositionInParent: function (node) {//*//
                            var parent = node.parent()[0];
                            var x_a = 0.0;

                            // Given node is not a direct child of the the root graph
                            if (parent != null) {
                                x_a = node.relativePosition('x') + (parent.width() / 2);
                            }
                            // Given node is a direct child of the the root graph

                            else {
                                x_a = node.position('x');
                            }

                            return x_a;
                        },
                        yPositionInParent: function (node) {//*//
                            var parent = node.parent()[0];

                            var y_a = 0.0;

                            // Given node is not a direct child of the the root graph
                            if (parent != null) {
                                y_a = node.relativePosition('y') + (parent.height() / 2);
                            }
                            // Given node is a direct child of the the root graph

                            else {
                                y_a = node.position('y');
                            }

                            return y_a;
                        },
                        /*
                         * for all children of the node parameter call this method
                         * with the same root parameter,
                         * remove the child and add the removed child to the collapsedchildren data
                         * of the root to restore them in the case of expandation
                         * root._private.data.collapsedChildren keeps the nodes to restore when the
                         * root is expanded
                         */
                        removeChildren: function (node, root) {
                            var children = node.children();
                            for (var i = 0; i < children.length; i++) {
                                var child = children[i];
                                this.removeChildren(child, root);
                                var parentData = cy.scratch('_cyExpandCollapse').parentData;
                                parentData[child.id()] = child.parent();
                                cy.scratch('_cyExpandCollapse').parentData = parentData;
                                var removedChild = child.remove();
                                if (root._private.data.collapsedChildren == null) {
                                    root._private.data.collapsedChildren = removedChild;
                                }
                                else {
                                    root._private.data.collapsedChildren = root._private.data.collapsedChildren.union(removedChild);
                                }
                            }
                        },
                        isMetaEdge: function (edge) {
                            return edge.hasClass("cy-expand-collapse-meta-edge");
                        },
                        barrowEdgesOfcollapsedChildren: function (node) {
                            var relatedNodes = node.descendants();
                            var edges = relatedNodes.edgesWith(cy.nodes().not(relatedNodes.union(node)));

                            var relatedNodeMap = {};

                            relatedNodes.each(function (ele, i) {
                                if (typeof ele === "number") {
                                    ele = i;
                                }
                                relatedNodeMap[ele.id()] = true;
                            });

                            for (var i = 0; i < edges.length; i++) {
                                var edge = edges[i];
                                var source = edge.source();
                                var target = edge.target();

                                if (!this.isMetaEdge(edge)) { // is original
                                    var originalEndsData = {
                                        source: source,
                                        target: target
                                    };

                                    edge.addClass("cy-expand-collapse-meta-edge");
                                    edge.data('originalEnds', originalEndsData);
                                }

                                edge.move({
                                    target: !relatedNodeMap[target.id()] ? target.id() : node.id(),
                                    source: !relatedNodeMap[source.id()] ? source.id() : node.id()
                                });
                            }
                        },
                        findNewEnd: function (node) {
                            var current = node;
                            var parentData = cy.scratch('_cyExpandCollapse').parentData;
                            var parent = parentData[current.id()];

                            while (!current.inside()) {
                                current = parent;
                                parent = parentData[parent.id()];
                            }

                            return current;
                        },
                        repairEdges: function (node) {
                            var connectedMetaEdges = node.connectedEdges('.cy-expand-collapse-meta-edge');

                            for (var i = 0; i < connectedMetaEdges.length; i++) {
                                var edge = connectedMetaEdges[i];
                                var originalEnds = edge.data('originalEnds');
                                var currentSrcId = edge.data('source');
                                var currentTgtId = edge.data('target');

                                if (currentSrcId === node.id()) {
                                    edge = edge.move({
                                        source: this.findNewEnd(originalEnds.source).id()
                                    });
                                } else {
                                    edge = edge.move({
                                        target: this.findNewEnd(originalEnds.target).id()
                                    });
                                }

                                if (edge.data('source') === originalEnds.source.id() && edge.data('target') === originalEnds.target.id()) {
                                    edge.removeClass('cy-expand-collapse-meta-edge');
                                    edge.removeData('originalEnds');
                                }
                            }
                        },
                        /*node is an outer node of root
                         if root is not it's anchestor
                         and it is not the root itself*/
                        isOuterNode: function (node, root) {//*//
                            var temp = node;
                            while (temp != null) {
                                if (temp == root) {
                                    return false;
                                }
                                temp = temp.parent()[0];
                            }
                            return true;
                        },
                        /**
                         * Get all collapsed children - including nested ones
                         * @param node : a collapsed node
                         * @param collapsedChildren : a collection to store the result
                         * @return : collapsed children
                         */
                        getCollapsedChildrenRecursively: function (node, collapsedChildren) {
                            var children = node.data('collapsedChildren') || [];
                            var i;
                            for (i = 0; i < children.length; i++) {
                                if (children[i].data('collapsedChildren')) {
                                    collapsedChildren = collapsedChildren.union(this.getCollapsedChildrenRecursively(children[i], collapsedChildren));
                                }
                                collapsedChildren = collapsedChildren.union(children[i]);
                            }
                            return collapsedChildren;
                        },
                        /* -------------------------------------- start section edge expand collapse -------------------------------------- */
                        collapseGivenEdges: function (edges, options) {
                            edges.unselect();
                            var nodes = edges.connectedNodes();
                            var edgesToCollapse = {};
                            // group edges by type if this option is set to true
                            if (options.groupEdgesOfSameTypeOnCollapse) {
                                edges.forEach(function (edge) {
                                    var edgeType = "unknown";
                                    if (options.edgeTypeInfo !== undefined) {
                                        edgeType = options.edgeTypeInfo instanceof Function ? options.edgeTypeInfo.call(edge) : edge.data()[options.edgeTypeInfo];
                                    }
                                    if (edgesToCollapse.hasOwnProperty(edgeType)) {
                                        edgesToCollapse[edgeType].edges = edgesToCollapse[edgeType].edges.add(edge);

                                        if (edgesToCollapse[edgeType].directionType == "unidirection" && (edgesToCollapse[edgeType].source != edge.source().id() || edgesToCollapse[edgeType].target != edge.target().id())) {
                                            edgesToCollapse[edgeType].directionType = "bidirection";
                                        }
                                    } else {
                                        var edgesX = cy.collection();
                                        edgesX = edgesX.add(edge);
                                        edgesToCollapse[edgeType] = { edges: edgesX, directionType: "unidirection", source: edge.source().id(), target: edge.target().id() }
                                    }
                                });
                            } else {
                                edgesToCollapse["unknown"] = { edges: edges, directionType: "unidirection", source: edges[0].source().id(), target: edges[0].target().id() }
                                for (var i = 0; i < edges.length; i++) {
                                    if (edgesToCollapse["unknown"].directionType == "unidirection" && (edgesToCollapse["unknown"].source != edges[i].source().id() || edgesToCollapse["unknown"].target != edges[i].target().id())) {
                                        edgesToCollapse["unknown"].directionType = "bidirection";
                                        break;
                                    }
                                }
                            }

                            var result = { edges: cy.collection(), oldEdges: cy.collection() }
                            var newEdges = [];
                            for (const edgeGroupType in edgesToCollapse) {
                                if (edgesToCollapse[edgeGroupType].edges.length < 2) {
                                    continue;
                                }
                                edges.trigger('expandcollapse.beforecollapseedge');
                                result.oldEdges = result.oldEdges.add(edgesToCollapse[edgeGroupType].edges);
                                var newEdge = {};
                                newEdge.group = "edges";
                                newEdge.data = {};
                                newEdge.data.source = edgesToCollapse[edgeGroupType].source;
                                newEdge.data.target = edgesToCollapse[edgeGroupType].target;
                                newEdge.data.id = "collapsedEdge_" + nodes[0].id() + "_" + nodes[1].id() + "_" + edgeGroupType + "_" + Math.floor(Math.random() * Date.now());
                                newEdge.data.collapsedEdges = cy.collection();

                                edgesToCollapse[edgeGroupType].edges.forEach(function (edge) {
                                    newEdge.data.collapsedEdges = newEdge.data.collapsedEdges.add(edge);
                                });

                                newEdge.data.collapsedEdges = this.check4nestedCollapse(newEdge.data.collapsedEdges, options);

                                var edgesTypeField = "edgeType";
                                if (options.edgeTypeInfo !== undefined) {
                                    edgesTypeField = options.edgeTypeInfo instanceof Function ? edgeTypeField : options.edgeTypeInfo;
                                }
                                newEdge.data[edgesTypeField] = edgeGroupType;

                                newEdge.data["directionType"] = edgesToCollapse[edgeGroupType].directionType;
                                newEdge.classes = "cy-expand-collapse-collapsed-edge";

                                newEdges.push(newEdge);
                                cy.remove(edgesToCollapse[edgeGroupType].edges);
                                edges.trigger('expandcollapse.aftercollapseedge');
                            }

                            result.edges = cy.add(newEdges);
                            return result;
                        },

                        check4nestedCollapse: function (edges2collapse, options) {
                            if (options.allowNestedEdgeCollapse) {
                                return edges2collapse;
                            }
                            let r = cy.collection();
                            for (let i = 0; i < edges2collapse.length; i++) {
                                let curr = edges2collapse[i];
                                let collapsedEdges = curr.data('collapsedEdges');
                                if (collapsedEdges && collapsedEdges.length > 0) {
                                    r = r.add(collapsedEdges);
                                } else {
                                    r = r.add(curr);
                                }
                            }
                            return r;
                        },

                        expandEdge: function (edge) {
                            edge.unselect();
                            var result = { edges: cy.collection(), oldEdges: cy.collection() }
                            var edges = edge.data('collapsedEdges');
                            if (edges !== undefined && edges.length > 0) {
                                edge.trigger('expandcollapse.beforeexpandedge');
                                result.oldEdges = result.oldEdges.add(edge);
                                cy.remove(edge);
                                result.edges = cy.add(edges);
                                edge.trigger('expandcollapse.afterexpandedge');
                            }
                            return result;
                        },

                        //if the edges are only between two nodes (valid for collpasing) returns the two nodes else it returns false
                        isValidEdgesForCollapse: function (edges) {
                            var endPoints = this.getEdgesDistinctEndPoints(edges);
                            if (endPoints.length != 2) {
                                return false;
                            } else {
                                return endPoints;
                            }
                        },

                        //returns a list of distinct endpoints of a set of edges.
                        getEdgesDistinctEndPoints: function (edges) {
                            var endPoints = [];
                            edges.forEach(function (edge) {
                                if (!this.containsElement(endPoints, edge.source())) {
                                    endPoints.push(edge.source());
                                }
                                if (!this.containsElement(endPoints, edge.target())) {
                                    endPoints.push(edge.target());

                                }
                            }.bind(this));

                            return endPoints;
                        },

                        //function to check if a list of elements contains the given element by looking at id()
                        containsElement: function (elements, element) {
                            var exists = false;
                            for (var i = 0; i < elements.length; i++) {
                                if (elements[i].id() == element.id()) {
                                    exists = true;
                                    break;
                                }
                            }
                            return exists;
                        }
                        /* -------------------------------------- end section edge expand collapse -------------------------------------- */
                    }

                };

                module.exports = expandCollapseUtilities;

            }, { "./boundingBoxUtilities": 1, "./elementUtilities": 5 }], 7: [function (_dereq_, module, exports) {
                ;
                (function () {
                    'use strict';

                    // registers the extension on a cytoscape lib ref
                    var register = function (cytoscape) {

                        if (!cytoscape) {
                            return;
                        } // can't register if cytoscape unspecified

                        var undoRedoUtilities = _dereq_('./undoRedoUtilities');
                        var cueUtilities = _dereq_("./cueUtilities");

                        function extendOptions(options, extendBy) {
                            var tempOpts = {};
                            for (var key in options)
                                tempOpts[key] = options[key];

                            for (var key in extendBy)
                                if (tempOpts.hasOwnProperty(key))
                                    tempOpts[key] = extendBy[key];
                            return tempOpts;
                        }

                        // evaluate some specific options in case of they are specified as functions to be dynamically changed
                        function evalOptions(options) {
                            var animate = typeof options.animate === 'function' ? options.animate.call() : options.animate;
                            var fisheye = typeof options.fisheye === 'function' ? options.fisheye.call() : options.fisheye;

                            options.animate = animate;
                            options.fisheye = fisheye;
                        }

                        // creates and returns the API instance for the extension
                        function createExtensionAPI(cy, expandCollapseUtilities) {
                            var api = {}; // API to be returned
                            // set functions

                            function handleNewOptions(opts) {
                                var currentOpts = getScratch(cy, 'options');
                                if (opts.cueEnabled && !currentOpts.cueEnabled) {
                                    api.enableCue();
                                }
                                else if (!opts.cueEnabled && currentOpts.cueEnabled) {
                                    api.disableCue();
                                }
                            }

                            // set all options at once
                            api.setOptions = function (opts) {
                                handleNewOptions(opts);
                                setScratch(cy, 'options', opts);
                            };

                            api.extendOptions = function (opts) {
                                var options = getScratch(cy, 'options');
                                var newOptions = extendOptions(options, opts);
                                handleNewOptions(newOptions);
                                setScratch(cy, 'options', newOptions);
                            }

                            // set the option whose name is given
                            api.setOption = function (name, value) {
                                var opts = {};
                                opts[name] = value;

                                var options = getScratch(cy, 'options');
                                var newOptions = extendOptions(options, opts);

                                handleNewOptions(newOptions);
                                setScratch(cy, 'options', newOptions);
                            };

                            // Collection functions

                            // collapse given eles extend options with given param
                            api.collapse = function (_eles, opts) {
                                var eles = this.collapsibleNodes(_eles);
                                var options = getScratch(cy, 'options');
                                var tempOptions = extendOptions(options, opts);
                                evalOptions(tempOptions);

                                return expandCollapseUtilities.collapseGivenNodes(eles, tempOptions);
                            };

                            // collapse given eles recursively extend options with given param
                            api.collapseRecursively = function (_eles, opts) {
                                var eles = this.collapsibleNodes(_eles);
                                var options = getScratch(cy, 'options');
                                var tempOptions = extendOptions(options, opts);
                                evalOptions(tempOptions);

                                return this.collapse(eles.union(eles.descendants()), tempOptions);
                            };

                            // expand given eles extend options with given param
                            api.expand = function (_eles, opts) {
                                var eles = this.expandableNodes(_eles);
                                var options = getScratch(cy, 'options');
                                var tempOptions = extendOptions(options, opts);
                                evalOptions(tempOptions);

                                return expandCollapseUtilities.expandGivenNodes(eles, tempOptions);
                            };

                            // expand given eles recusively extend options with given param
                            api.expandRecursively = function (_eles, opts) {
                                var eles = this.expandableNodes(_eles);
                                var options = getScratch(cy, 'options');
                                var tempOptions = extendOptions(options, opts);
                                evalOptions(tempOptions);

                                return expandCollapseUtilities.expandAllNodes(eles, tempOptions);
                            };


                            // Core functions

                            // collapse all collapsible nodes
                            api.collapseAll = function (opts) {
                                var options = getScratch(cy, 'options');
                                var tempOptions = extendOptions(options, opts);
                                evalOptions(tempOptions);

                                return this.collapseRecursively(this.collapsibleNodes(), tempOptions);
                            };

                            // expand all expandable nodes
                            api.expandAll = function (opts) {
                                var options = getScratch(cy, 'options');
                                var tempOptions = extendOptions(options, opts);
                                evalOptions(tempOptions);

                                return this.expandRecursively(this.expandableNodes(), tempOptions);
                            };


                            // Utility functions

                            // returns if the given node is expandable
                            api.isExpandable = function (node) {
                                return node.hasClass('cy-expand-collapse-collapsed-node');
                            };

                            // returns if the given node is collapsible
                            api.isCollapsible = function (node) {
                                return !this.isExpandable(node) && node.isParent();
                            };

                            // get collapsible ones inside given nodes if nodes parameter is not specified consider all nodes
                            api.collapsibleNodes = function (_nodes) {
                                var self = this;
                                var nodes = _nodes ? _nodes : cy.nodes();
                                return nodes.filter(function (ele, i) {
                                    if (typeof ele === "number") {
                                        ele = i;
                                    }
                                    return self.isCollapsible(ele);
                                });
                            };

                            // get expandable ones inside given nodes if nodes parameter is not specified consider all nodes
                            api.expandableNodes = function (_nodes) {
                                var self = this;
                                var nodes = _nodes ? _nodes : cy.nodes();
                                return nodes.filter(function (ele, i) {
                                    if (typeof ele === "number") {
                                        ele = i;
                                    }
                                    return self.isExpandable(ele);
                                });
                            };

                            // Get the children of the given collapsed node which are removed during collapse operation
                            api.getCollapsedChildren = function (node) {
                                return node.data('collapsedChildren');
                            };

                            /** Get collapsed children recursively including nested collapsed children
                             * Returned value includes edges and nodes, use selector to get edges or nodes
                             * @param node : a collapsed node
                             * @return all collapsed children
                             */
                            api.getCollapsedChildrenRecursively = function (node) {
                                var collapsedChildren = cy.collection();
                                return expandCollapseUtilities.getCollapsedChildrenRecursively(node, collapsedChildren);
                            };

                            /** Get collapsed children of all collapsed nodes recursively including nested collapsed children
                             * Returned value includes edges and nodes, use selector to get edges or nodes
                             * @return all collapsed children
                             */
                            api.getAllCollapsedChildrenRecursively = function () {
                                var collapsedChildren = cy.collection();
                                var collapsedNodes = cy.nodes(".cy-expand-collapse-collapsed-node");
                                var j;
                                for (j = 0; j < collapsedNodes.length; j++) {
                                    collapsedChildren = collapsedChildren.union(this.getCollapsedChildrenRecursively(collapsedNodes[j]));
                                }
                                return collapsedChildren;
                            };
                            // This method forces the visual cue to be cleared. It is to be called in extreme cases
                            api.clearVisualCue = function (node) {
                                cy.trigger('expandcollapse.clearvisualcue');
                            };

                            api.disableCue = function () {
                                var options = getScratch(cy, 'options');
                                if (options.cueEnabled) {
                                    cueUtilities('unbind', cy, api);
                                    options.cueEnabled = false;
                                }
                            };

                            api.enableCue = function () {
                                var options = getScratch(cy, 'options');
                                if (!options.cueEnabled) {
                                    cueUtilities('rebind', cy, api);
                                    options.cueEnabled = true;
                                }
                            };

                            api.getParent = function (nodeId) {
                                if (cy.getElementById(nodeId)[0] === undefined) {
                                    var parentData = getScratch(cy, 'parentData');
                                    return parentData[nodeId];
                                }
                                else {
                                    return cy.getElementById(nodeId).parent();
                                }
                            };

                            api.collapseEdges = function (edges, opts) {
                                var result = { edges: cy.collection(), oldEdges: cy.collection() };
                                if (edges.length < 2) return result;
                                if (edges.connectedNodes().length > 2) return result;
                                var options = getScratch(cy, 'options');
                                var tempOptions = extendOptions(options, opts);
                                return expandCollapseUtilities.collapseGivenEdges(edges, tempOptions);
                            };
                            api.expandEdges = function (edges) {
                                var result = { edges: cy.collection(), oldEdges: cy.collection() }
                                if (edges === undefined) return result;

                                //if(typeof edges[Symbol.iterator] === 'function'){//collection of edges is passed
                                edges.forEach(function (edge) {
                                    var operationResult = expandCollapseUtilities.expandEdge(edge);
                                    result.edges = result.edges.add(operationResult.edges);
                                    result.oldEdges = result.oldEdges.add(operationResult.oldEdges);

                                });
                                /*  }else{//one edge passed
                                   var operationResult = expandCollapseUtilities.expandEdge(edges);
                                   result.edges = result.edges.add(operationResult.edges);
                                   result.oldEdges = result.oldEdges.add(operationResult.oldEdges);

                                 } */

                                return result;

                            };
                            api.collapseEdgesBetweenNodes = function (nodes, opts) {
                                var options = getScratch(cy, 'options');
                                var tempOptions = extendOptions(options, opts);
                                function pairwise(list) {
                                    var pairs = [];
                                    list
                                        .slice(0, list.length - 1)
                                        .forEach(function (first, n) {
                                            var tail = list.slice(n + 1, list.length);
                                            tail.forEach(function (item) {
                                                pairs.push([first, item])
                                            });
                                        })
                                    return pairs;
                                }
                                var nodesPairs = pairwise(nodes);
                                var result = { edges: cy.collection(), oldEdges: cy.collection() };
                                nodesPairs.forEach(function (nodePair) {
                                    var edges = nodePair[0].connectedEdges('[source = "' + nodePair[1].id() + '"],[target = "' + nodePair[1].id() + '"]');

                                    if (edges.length >= 2) {
                                        var operationResult = expandCollapseUtilities.collapseGivenEdges(edges, tempOptions)
                                        result.oldEdges = result.oldEdges.add(operationResult.oldEdges);
                                        result.edges = result.edges.add(operationResult.edges);
                                    }

                                }.bind(this));

                                return result;

                            };
                            api.expandEdgesBetweenNodes = function (nodes) {
                                if (nodes.length <= 1) cy.collection();
                                var edgesToExpand = cy.collection();
                                function pairwise(list) {
                                    var pairs = [];
                                    list
                                        .slice(0, list.length - 1)
                                        .forEach(function (first, n) {
                                            var tail = list.slice(n + 1, list.length);
                                            tail.forEach(function (item) {
                                                pairs.push([first, item])
                                            });
                                        })
                                    return pairs;
                                }
                                //var result = {edges: cy.collection(), oldEdges: cy.collection()}   ;
                                var nodesPairs = pairwise(nodes);
                                nodesPairs.forEach(function (nodePair) {
                                    var edges = nodePair[0].connectedEdges('.cy-expand-collapse-collapsed-edge[source = "' + nodePair[1].id() + '"],[target = "' + nodePair[1].id() + '"]');
                                    edgesToExpand = edgesToExpand.union(edges);

                                }.bind(this));
                                //result.oldEdges = result.oldEdges.add(edgesToExpand);
                                //result.edges = result.edges.add(this.expandEdges(edgesToExpand));
                                return this.expandEdges(edgesToExpand);
                            };
                            api.collapseAllEdges = function (opts) {
                                var options = getScratch(cy, 'options');
                                var tempOptions = extendOptions(options, opts);
                                function pairwise(list) {
                                    var pairs = [];
                                    list
                                        .slice(0, list.length - 1)
                                        .forEach(function (first, n) {
                                            var tail = list.slice(n + 1, list.length);
                                            tail.forEach(function (item) {
                                                pairs.push([first, item])
                                            });
                                        })
                                    return pairs;
                                }

                                return this.collapseEdgesBetweenNodes(cy.edges().connectedNodes(), opts);
                                /*  var nodesPairs = pairwise(cy.edges().connectedNodes());
                                 nodesPairs.forEach(function(nodePair){
                                   var edges = nodePair[0].connectedEdges('[source = "'+ nodePair[1].id()+'"],[target = "'+ nodePair[1].id()+'"]');
                                   if(edges.length >=2){
                                     expandCollapseUtilities.collapseGivenEdges(edges, tempOptions);
                                   }

                                 }.bind(this)); */

                            };
                            api.expandAllEdges = function () {
                                var edges = cy.edges(".cy-expand-collapse-collapsed-edge");
                                var result = { edges: cy.collection(), oldEdges: cy.collection() };
                                var operationResult = this.expandEdges(edges);
                                result.oldEdges = result.oldEdges.add(operationResult.oldEdges);
                                result.edges = result.edges.add(operationResult.edges);
                                return result;
                            };



                            return api; // Return the API instance
                        }

                        // Get the whole scratchpad reserved for this extension (on an element or core) or get a single property of it
                        function getScratch(cyOrEle, name) {
                            if (cyOrEle.scratch('_cyExpandCollapse') === undefined) {
                                cyOrEle.scratch('_cyExpandCollapse', {});
                            }

                            var scratch = cyOrEle.scratch('_cyExpandCollapse');
                            var retVal = (name === undefined) ? scratch : scratch[name];
                            return retVal;
                        }

                        // Set a single property on scratchpad of an element or the core
                        function setScratch(cyOrEle, name, val) {
                            getScratch(cyOrEle)[name] = val;
                        }

                        // register the extension cy.expandCollapse()
                        cytoscape("core", "expandCollapse", function (opts) {
                            var cy = this;

                            var options = getScratch(cy, 'options') || {
                                layoutBy: null, // for rearrange after expand/collapse. It's just layout options or whole layout function. Choose your side!
                                fisheye: true, // whether to perform fisheye view after expand/collapse you can specify a function too
                                animate: true, // whether to animate on drawing changes you can specify a function too
                                animationDuration: 1000, // when animate is true, the duration in milliseconds of the animation
                                ready: function () { }, // callback when expand/collapse initialized
                                undoable: true, // and if undoRedoExtension exists,

                                cueEnabled: true, // Whether cues are enabled
                                expandCollapseCuePosition: 'top-left', // default cue position is top left you can specify a function per node too
                                expandCollapseCueSize: 12, // size of expand-collapse cue
                                expandCollapseCueLineSize: 8, // size of lines used for drawing plus-minus icons
                                expandCueImage: undefined, // image of expand icon if undefined draw regular expand cue
                                collapseCueImage: undefined, // image of collapse icon if undefined draw regular collapse cue
                                expandCollapseCueSensitivity: 1, // sensitivity of expand-collapse cues

                                edgeTypeInfo: "edgeType", //the name of the field that has the edge type, retrieved from edge.data(), can be a function
                                groupEdgesOfSameTypeOnCollapse: false,
                                allowNestedEdgeCollapse: true,
                                zIndex: 999 // z-index value of the canvas in which cue ımages are drawn
                            };

                            // If opts is not 'get' that is it is a real options object then initilize the extension
                            if (opts !== 'get') {
                                options = extendOptions(options, opts);

                                var expandCollapseUtilities = _dereq_('./expandCollapseUtilities')(cy);
                                var api = createExtensionAPI(cy, expandCollapseUtilities); // creates and returns the API instance for the extension

                                setScratch(cy, 'api', api);

                                undoRedoUtilities(cy, api);

                                cueUtilities(options, cy, api);

                                // if the cue is not enabled unbind cue events
                                if (!options.cueEnabled) {
                                    cueUtilities('unbind', cy, api);
                                }

                                if (options.ready) {
                                    options.ready();
                                }

                                setScratch(cy, 'options', options);

                                var parentData = {};
                                setScratch(cy, 'parentData', parentData);
                            }

                            return getScratch(cy, 'api'); // Expose the API to the users
                        });
                    };


                    if (typeof module !== 'undefined' && module.exports) { // expose as a commonjs module
                        module.exports = register;
                    }

                    if (typeof define !== 'undefined' && define.amd) { // expose as an amd/requirejs module
                        define('cytoscape-expand-collapse', function () {
                            return register;
                        });
                    }

                    if (typeof cytoscape !== 'undefined') { // expose to global cytoscape (i.e. window.cytoscape)
                        register(cytoscape);
                    }

                })();

            }, { "./cueUtilities": 2, "./expandCollapseUtilities": 6, "./undoRedoUtilities": 8 }], 8: [function (_dereq_, module, exports) {
                module.exports = function (cy, api) {
                    if (cy.undoRedo == null)
                        return;

                    var ur = cy.undoRedo({}, true);

                    function getEles(_eles) {
                        return (typeof _eles === "string") ? cy.$(_eles) : _eles;
                    }

                    function getNodePositions() {
                        var positions = {};
                        var nodes = cy.nodes();

                        for (var i = 0; i < nodes.length; i++) {
                            var ele = nodes[i];
                            positions[ele.id()] = {
                                x: ele.position("x"),
                                y: ele.position("y")
                            };
                        }

                        return positions;
                    }

                    function returnToPositions(positions) {
                        var currentPositions = {};
                        cy.nodes().not(":parent").positions(function (ele, i) {
                            if (typeof ele === "number") {
                                ele = i;
                            }
                            currentPositions[ele.id()] = {
                                x: ele.position("x"),
                                y: ele.position("y")
                            };
                            var pos = positions[ele.id()];
                            return {
                                x: pos.x,
                                y: pos.y
                            };
                        });

                        return currentPositions;
                    }

                    var secondTimeOpts = {
                        layoutBy: null,
                        animate: false,
                        fisheye: false
                    };

                    function doIt(func) {
                        return function (args) {
                            var result = {};
                            var nodes = getEles(args.nodes);
                            if (args.firstTime) {
                                result.oldData = getNodePositions();
                                result.nodes = func.indexOf("All") > 0 ? api[func](args.options) : api[func](nodes, args.options);
                            } else {
                                result.oldData = getNodePositions();
                                result.nodes = func.indexOf("All") > 0 ? api[func](secondTimeOpts) : api[func](cy.collection(nodes), secondTimeOpts);
                                returnToPositions(args.oldData);
                            }

                            return result;
                        };
                    }

                    var actions = ["collapse", "collapseRecursively", "collapseAll", "expand", "expandRecursively", "expandAll"];

                    for (var i = 0; i < actions.length; i++) {
                        if (i == 2)
                            ur.action("collapseAll", doIt("collapseAll"), doIt("expandRecursively"));
                        else if (i == 5)
                            ur.action("expandAll", doIt("expandAll"), doIt("collapseRecursively"));
                        else
                            ur.action(actions[i], doIt(actions[i]), doIt(actions[(i + 3) % 6]));
                    }

                    function collapseEdges(args) {
                        var options = args.options;
                        var edges = args.edges;
                        var result = {};

                        result.options = options;
                        if (args.firstTime) {
                            var collapseResult = api.collapseEdges(edges, options);
                            result.edges = collapseResult.edges;
                            result.oldEdges = collapseResult.oldEdges;
                            result.firstTime = false;
                        } else {
                            result.oldEdges = edges;
                            result.edges = args.oldEdges;
                            if (args.edges.length > 0 && args.oldEdges.length > 0) {
                                cy.remove(args.edges);
                                cy.add(args.oldEdges);
                            }


                        }

                        return result;
                    }
                    function collapseEdgesBetweenNodes(args) {
                        var options = args.options;
                        var result = {};
                        result.options = options;
                        if (args.firstTime) {
                            var collapseAllResult = api.collapseEdgesBetweenNodes(args.nodes, options);
                            result.edges = collapseAllResult.edges;
                            result.oldEdges = collapseAllResult.oldEdges;
                            result.firstTime = false;
                        } else {
                            result.edges = args.oldEdges;
                            result.oldEdges = args.edges;
                            if (args.edges.length > 0 && args.oldEdges.length > 0) {
                                cy.remove(args.edges);
                                cy.add(args.oldEdges);
                            }

                        }

                        return result;

                    }
                    function collapseAllEdges(args) {
                        var options = args.options;
                        var result = {};
                        result.options = options;
                        if (args.firstTime) {
                            var collapseAllResult = api.collapseAllEdges(options);
                            result.edges = collapseAllResult.edges;
                            result.oldEdges = collapseAllResult.oldEdges;
                            result.firstTime = false;
                        } else {
                            result.edges = args.oldEdges;
                            result.oldEdges = args.edges;
                            if (args.edges.length > 0 && args.oldEdges.length > 0) {
                                cy.remove(args.edges);
                                cy.add(args.oldEdges);
                            }

                        }

                        return result;
                    }
                    function expandEdges(args) {
                        var options = args.options;
                        var result = {};

                        result.options = options;
                        if (args.firstTime) {
                            var expandResult = api.expandEdges(args.edges);
                            result.edges = expandResult.edges;
                            result.oldEdges = expandResult.oldEdges;
                            result.firstTime = false;

                        } else {
                            result.oldEdges = args.edges;
                            result.edges = args.oldEdges;
                            if (args.edges.length > 0 && args.oldEdges.length > 0) {
                                cy.remove(args.edges);
                                cy.add(args.oldEdges);
                            }

                        }

                        return result;
                    }
                    function expandEdgesBetweenNodes(args) {
                        var options = args.options;
                        var result = {};
                        result.options = options;
                        if (args.firstTime) {
                            var collapseAllResult = api.expandEdgesBetweenNodes(args.nodes, options);
                            result.edges = collapseAllResult.edges;
                            result.oldEdges = collapseAllResult.oldEdges;
                            result.firstTime = false;
                        } else {
                            result.edges = args.oldEdges;
                            result.oldEdges = args.edges;
                            if (args.edges.length > 0 && args.oldEdges.length > 0) {
                                cy.remove(args.edges);
                                cy.add(args.oldEdges);
                            }

                        }

                        return result;
                    }
                    function expandAllEdges(args) {
                        var options = args.options;
                        var result = {};
                        result.options = options;
                        if (args.firstTime) {
                            var expandResult = api.expandAllEdges(options);
                            result.edges = expandResult.edges;
                            result.oldEdges = expandResult.oldEdges;
                            result.firstTime = false;
                        } else {
                            result.edges = args.oldEdges;
                            result.oldEdges = args.edges;
                            if (args.edges.length > 0 && args.oldEdges.length > 0) {
                                cy.remove(args.edges);
                                cy.add(args.oldEdges);
                            }

                        }

                        return result;
                    }


                    ur.action("collapseEdges", collapseEdges, expandEdges);
                    ur.action("expandEdges", expandEdges, collapseEdges);

                    ur.action("collapseEdgesBetweenNodes", collapseEdgesBetweenNodes, expandEdgesBetweenNodes);
                    ur.action("expandEdgesBetweenNodes", expandEdgesBetweenNodes, collapseEdgesBetweenNodes);

                    ur.action("collapseAllEdges", collapseAllEdges, expandAllEdges);
                    ur.action("expandAllEdges", expandAllEdges, collapseAllEdges);







                };

            }, {}]
        }, {}, [7])(7)
    });

        document.addEventListener('DOMContentLoaded', function () {

            var elements = _________ELEMENTS_GO_HERE_________

            var cy = window.cy = cytoscape({
                container: document.getElementById('cy'),

                layout: {
                    name: 'cose-bilkent'
                },

                style: [
                    {
                        selector: 'node',
                        style: {
                            'background-color': '#e89d90',
                            'label': 'data(label)',
                        }
                    },
                    {
                        selector: ':parent',
                        style: {
                            'background-opacity': 0.333
                        }
                    },

                    {
                        selector: "node.cy-expand-collapse-collapsed-node",
                        style: {
                            "background-color": "#ee4c2c",
                            "shape": "rectangle"
                        }
                    },

                    {
                        selector: 'edge',
                        style: {
                            'width': 1,
                            'line-color': '#979797',
                            'curve-style': 'straight'
                        }
                    },
                    {
                        "selector": "edge[arrow]",
                        "style": {
                            "target-arrow-shape": "data(arrow)"
                        }
                    },
                    {
                        selector: 'edge.meta',
                        style: {
                            'width': 2,
                            'line-color': 'red'
                        }
                    },

                    {
                        selector: ':selected',
                        style: {
                            "border-width": 3,
                            "border-color": '#DAA520'
                        }
                    }
                ],
                elements
            });

            var api = cy.expandCollapse({
                layoutBy: {
                    name: "cose-bilkent",
                    animate: "end",
                    randomize: false,
                    fit: true
                },
                fisheye: true,
                animate: true,
                undoable: false,
            });

            cy.nodes().forEach(function (ele) {
                if (api.isCollapsible(ele)) {
                    api.collapseRecursively(ele);
                }
            });


            document.addEventListener('keyup', (e) => {
                if (e.code == "Space") {
                    if (cy.$(":selected").hasClass('cy-expand-collapse-collapsed-node')) {
                        api.expandRecursively(cy.$(":selected"));
                    } else {
                        if (api.isCollapsible(cy.$(":selected"))) {
                            api.collapseRecursively(cy.$(":selected"));
                        } else {
                            node = cy.$(":selected");

                            var scratch = cy.scratch("_hidden." + node.id());

                            if (scratch == null) {

                                var component = cy.collection();
                                node.successors().forEach(function (ele) {
                                    component.push(ele);
                                });

                                // Get the connected component of the selected node.
                                // var component = node.component();
                                var exclude = cy.collection();
                                exclude.push(node);

                                // Get only the nodes from the connected component.
                                var componentNodes = component.filter(function (ele, i, eles) {
                                    return ele.isNode();
                                });

                                // Figure out which nodes from the connected component have
                                // incident edges that are NOT in the connected component.
                                // These should not be hidden.
                                componentNodes.filter(function (node) {
                                    // Iterate over incident edges.
                                    var outsideEdges = cy.collection();
                                    node.incomers().forEach(function (edge, i, eles) {
                                        // If the incident edge is not in the connected component,
                                        // do not mark the node for hiding.
                                        if (!edge.isNode() && component.has(edge) == false) {
                                            outsideEdges.push(edge);
                                        }
                                    });

                                    if (outsideEdges.length > 0) {
                                        exclude.push(node);
                                        outsideEdges.forEach(function (edge, i, eles) {
                                            exclude.push(edge);
                                        });
                                    }
                                });

                                var hideList = component.difference(exclude);

                                cy.nodes().filter(function (node) {
                                    if (node.isParent()) {
                                        var allChildrenInHideList = true;
                                        node.children().forEach(function (child) {
                                            if (hideList.has(child) == false) {
                                                allChildrenInHideList = false;
                                            }
                                        });

                                        if (allChildrenInHideList) {
                                            hideList.push(node)
                                            if (api.isCollapsible(node)) {
                                                api.collapseRecursively(node);
                                            }
                                        }
                                    }
                                })

                                component = cy.collection();
                                node.successors().forEach(function (ele) {
                                    component.push(ele);
                                });

                                hideList = component.difference(exclude);

                                cy.scratch("_hidden." + node.id(), hideList);
                                cy.remove(hideList);
                            } else {
                                var scratch = cy.scratch("_hidden." + node.id());
                                cy.add(scratch)
                                cy.scratch("_hidden." + node.id(), null);
                            }
                        }
                    }
                }
            });
        });
    </script>
</head>

<body>
    <div >
    <h1>torch.package Dependency Visualizer</h1>
    <b>Space - expand/collapse selected module, show/hide module dependencies</b>
    </div>
    <div id="cy"></div>

</body>

</html>
"""
