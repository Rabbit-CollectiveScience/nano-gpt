import * as d3 from 'd3';

export function drawAttentionHeatmap(container, tensorData) {
  // tensorData is expected to be nested arrays: [heads, T, T] 
  const containerSelection = d3.select(container);
  
  // Wipe canvas before every animation frame
  containerSelection.selectAll('*').remove();

  if (!tensorData || tensorData.length === 0) return;

  const n_heads = tensorData.length;
  const T = tensorData[0].length;
  
  // Matrix Box Parameters
  const margin = {top: 30, right: 30, bottom: 30, left: 30};
  const cellSize = 30; // 30px per mathematical square
  const gridWidth = T * cellSize;
  const gridHeight = T * cellSize;
  const xOffset = gridWidth + 40; // Gaps between Attention Heads
  
  // Total dimensions required by SVG canvas
  const totalWidth = margin.left + (xOffset * n_heads) + margin.right;
  const totalHeight = margin.top + gridHeight + margin.bottom;

  // Render the core Canvas Engine
  const svg = containerSelection.append('svg')
    .attr('width', totalWidth)
    .attr('height', totalHeight)
    .attr('viewBox', `0 0 ${totalWidth} ${totalHeight}`)
    .style('max-width', '100%')
    .style('max-height', '100%');

  // Load the mathematically proven Inferno colormap 
  const colorScale = d3.scaleSequential(d3.interpolateInferno)
    .domain([0, 1]); // Softmax boundaries (0.0 to 1.0)

  // Subdivide canvas into isolated groups for each Attention Head
  tensorData.forEach((headMatrix, headIdx) => {
    const headGroup = svg.append('g')
      .attr('transform', `translate(${margin.left + headIdx * xOffset}, ${margin.top})`);

    // Affix Label header
    headGroup.append('text')
      .text(`Attention Head ${headIdx}`)
      .attr('x', gridWidth / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ccc')
      .style('font-family', 'sans-serif')
      .style('font-size', '14px');

    // Recursively draw tensor properties via SVG vectors
    for (let r = 0; r < T; r++) {
      for (let c = 0; c < T; c++) {
        const val = headMatrix[r][c];
        
        const cell = headGroup.append('g')
          .attr('class', 'cell')
          .attr('transform', `translate(${c * cellSize}, ${r * cellSize})`);
          
        // Background matrix weight intensity
        cell.append('rect')
          .attr('width', cellSize)
          .attr('height', cellSize)
          .attr('fill', Number.isNaN(val) || val < 0 ? '#111' : colorScale(val))
          .attr('stroke', '#333')
          .attr('stroke-width', 1)
          .attr('rx', 2);
          
        // Inject matrix value string (excluding masked items)
        if (!Number.isNaN(val) && val >= 0.01) {
          cell.append('text')
            .text(val.toFixed(2))
            .attr('x', cellSize / 2)
            .attr('y', cellSize / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', val > 0.5 ? '#fff' : '#ccc');
        }
      }
    }
  });
}
