import React, { useEffect, useCallback } from "react"
import $ from "jquery"

const PatternSelect = ({letters, onClear, setValue, onSubmit}) => {
    const moveEvent = 'touchstart mousedown touchmove mousemove'
    const svgns = 'http://www.w3.org/2000/svg'
  
    const PatternLock = useCallback((element) => {
        const scrollKeys = {
            37: true, // left
            38: true, // up
            39: true, // right
            40: true, // down
            32: true, // spacebar
            33: true, // pageup
            34: true, // pagedown
            35: true, // end
            36: true, // home
        };
        
        let svg = $(element)
        let root = svg[0]
        let dots = svg.find('.lock-dots text')
        let lines = svg.find('.lock-lines')
        let actives = svg.find('.lock-actives')
        const pt = root.createSVGPoint();
        let code = []
        let currentline
        let currenthandler

        svg.on('touchstart mousedown', (e) => {
            clear()
            e.preventDefault()
            disableScroll()
            discoverDot(e)
            svg.on(moveEvent, discoverDot)
            let endEvent = e.type === 'touchstart' ? 'touchend' : 'mouseup';
            $(document).one(endEvent, (e) => {
                end()
            })
        })

        function getPattern() {
            const pattern = code.map((i) => i.attributes.value.value).join('')
            return pattern
        }

        async function end() {
            enableScroll()
            stopTrack(currentline)
            currentline && currentline.remove()
            svg.off(moveEvent, discoverDot)
            const pattern = getPattern()
            await setValue(pattern)
            await onSubmit(pattern)
            await clear();
            for (let i = 0; i < letters.length; i++){
                document.getElementById(`letter_${i}`).setAttribute("fill", "var(--c-tertiary)")
            }
            
            // console.log(pattern)
        }

        function clear() {
            code = []
            currentline = undefined
            currenthandler = undefined
            svg.removeClass('success error')
            lines.empty()
            actives.empty()
            onClear()
        }

        function preventDefault(e) {
            e.preventDefault();

        }

        function preventDefaultForScrollKeys(e) {
            if (scrollKeys[e.keyCode]) {
                preventDefault(e);
                return false;
            }
        }

        function disableScroll() {
            if (window.addEventListener) // older FF
                window.addEventListener('DOMMouseScroll', preventDefault, false);
            window.onwheel = preventDefault; // modern standard
            window.onmousewheel = document.onmousewheel = preventDefault; // older browsers, IE
            window.ontouchmove = preventDefault; // mobile
            document.onkeydown = preventDefaultForScrollKeys;
        }

        function enableScroll() {
            if (window.removeEventListener)
                window.removeEventListener('DOMMouseScroll', preventDefault, false);
            window.onmousewheel = document.onmousewheel = null;
            window.onwheel = null;
            window.ontouchmove = null;
            document.onkeydown = null;
        }

        function isUsed(target) {
            for (let i = 0; i < code.length; i++) {
                if (code[i] === target) {
                    return true
                }
            }
            return false
        }

        function isAvailable(target) {
            for (let i = 0; i < dots.length; i++) {
                if (dots[i] === target) {
                    return true
                }
            }
            return false
        }

        function updateLine(line) {
            return function(e) {
                e.preventDefault()
                if (currentline !== line) return
                let pos = svgPosition(e.target, e)
                line.setAttribute('x2', pos.x)
                line.setAttribute('y2', pos.y)
                return false
            }
        }

        function discoverDot(e, target) {
            if (!target) {
                let {x, y} = getMousePos(e)
                target = document.elementFromPoint(x, y);
            }
            if (isAvailable(target) && !isUsed(target)) {
                stopTrack(currentline, target)
                currentline = beginTrack(target)
            }
        }

        function stopTrack(line, target) {
            if (line === undefined) return
            if (currenthandler) {
                svg.off('touchmove mousemove', currenthandler)
            }
            if (target === undefined) return
            let x = target.getAttribute('cx') || target.getAttribute("x")
            let y = target.getAttribute('cy') || target.getAttribute("y")
            line.setAttribute('x2', x)
            line.setAttribute('y2', y)
        }

        function beginTrack(target) { 
            code.push(target)
            setValue(code?.map((i) => i.attributes.value.value).join(''))
            let x = target.getAttribute('cx') || target.getAttribute("x")
            let y = target.getAttribute('cy') || target.getAttribute("y")
            const line = createNewLine(x, y)
            const marker = createNewMarker(x, y)
            document.getElementById(target.getAttribute("id")).setAttribute("fill", "var(--c-primary)")
            actives.append(marker)
            currenthandler = updateLine(line)
            svg.on('touchmove mousemove', currenthandler)
            lines.append(line);
            return line
        }

        function createNewMarker(x, y) {
            const marker = document.createElementNS(svgns, "circle")
            marker.setAttribute('cx', x)
            marker.setAttribute('cy', y)
            marker.setAttribute('r', 10)
            
            return marker
        }

        function createNewLine(x1, y1, x2, y2) {
            const line = document.createElementNS(svgns, "line")
            line.setAttribute('x1', x1)
            line.setAttribute('y1', y1)
            if (x2 === undefined || y2 === undefined) {
                line.setAttribute('x2', x1)
                line.setAttribute('y2', y1)
            } else {
                line.setAttribute('x2', x2)
                line.setAttribute('y2', y2)
            }
            return line
        }

        function getMousePos(e) {
            return {
                x: e.clientX || e?.originalEvent?.touches[0]?.clientX || 0,
                y :e.clientY || e?.originalEvent?.touches[0]?.clientY || 0
            }
        }

        function svgPosition(element, e) {
            let {x, y} = getMousePos(e)
            pt.x = x; pt.y = y;
            return pt.matrixTransform(element.getScreenCTM().inverse());
        }

    
    }, [onClear, onSubmit, letters.length, setValue])
    //eslint-disable-next-line react-hooks/exhaustive-deps
    useEffect(() => PatternLock("#lock") , [])

    const getCoordinates = (index) => {
        const alpha = 2 * Math.PI * index / letters.length 
        const x =  50 + 39 * Math.cos(alpha)
        const y = 50 + 39 * Math.sin(alpha)
        return {x: x, y: y}
    }

    return (
        
    <div style={{display: "flex", justifyContent:"center", padding: 20}}>
        <svg style={{width: 225, height: 225, background: "var(--c-secondary)", borderRadius: 112.5}} className="patternlock" id="lock" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <g className="lock-actives"></g>
                <g className="lock-lines"></g>
                <g className="lock-dots">
                    {letters && letters.length && letters.map((l,i) => {
                        return(
                            <React.Fragment key={i}>
                                <circle index={i} data={l.l} cx={getCoordinates(i).x} cy={getCoordinates(i).y} r={6.5} opacity="0" />
                                <text className="svgtxt" value={l.l} id={`letter_${i}`} textAnchor="middle" x={getCoordinates(i).x} y={getCoordinates(i).y} fill="var(--c-tertiary)" fontSize="0.8em"  dy=".3em">{l.l.toUpperCase()}</text>
                            </React.Fragment>
                        )}
                    )}
                    
                </g>
            </svg>
        </div>
    )

}
export default PatternSelect
