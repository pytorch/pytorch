function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}

export function getColorForCallstack(callstack) {
  // Use the last line of callstack for color
  const lastLine = callstack[callstack.length - 1] || '';
  const hash = hashString(lastLine);

  // Generate HSL color with good saturation and lightness
  const hue = hash % 360;
  const saturation = 65 + (hash % 20);
  const lightness = 60 + (hash % 15);

  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

export function getDarkerColor(color) {
  // Parse HSL and darken
  const match = color.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
  if (match) {
    const [, h, s, l] = match;
    return `hsl(${h}, ${s}%, ${Math.max(0, parseInt(l) - 15)}%)`;
  }
  return color;
}
