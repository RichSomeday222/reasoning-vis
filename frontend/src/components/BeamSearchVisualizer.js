// src/components/BeamSearchVisualizer.js
import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { mockTreeData } from '../data/mockTreeData'; 


const BeamSearchVisualizer = () => {
  const svgRef = useRef();
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedPath, setSelectedPath] = useState(null);
  const [highlightedPath, setHighlightedPath] = useState(null);
  const [collapsedNodes, setCollapsedNodes] = useState(new Set());

  const getQualityColor = (score) => {
    if (score >= 0.9) return '#166534'; 
    if (score >= 0.7) return '#10b981'; 
    if (score >= 0.5) return '#86efac'; 
    return '#EF4444'; 
  };

  const getReasoningColor = (type) => {
    const colors = {
      'start': '#6B7280',
      'problem_understanding': '#8B5CF6',
      'calculation': '#06B6D4',
      'conclusion': '#10B981'
    };
    return colors[type] || '#6B7280';
  };

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 1200;
    const height = 800;
    const margin = {top: 50, right: 50, bottom: 50, left: 50};

    // 构建层次结构
    // 构建层次数据
const buildHierarchy = () => {
  const nodes = mockTreeData.beam_tree;
  
  const buildNode = (nodeId) => {
      const node = nodes[nodeId];
      // 如果节点被折叠，不显示其子节点
      if (collapsedNodes.has(nodeId)) {
        return {
          id: nodeId,
          data: { ...node, _collapsed: true },
          children: []
        };
      }
      
      return {
        id: nodeId,
        data: { ...node, _collapsed: false },
        children: node.children.map(childId => buildNode(childId))
      };
    };
    
    return buildNode('root');
  };

    const hierarchyData = buildHierarchy();
    const treeLayout = d3.tree().size([width - margin.left - margin.right, height - margin.top - margin.bottom]);
    const root = d3.hierarchy(hierarchyData);
    treeLayout(root);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // 绘制连接线
    const links = g.selectAll('.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkVertical()
        .x(d => d.x)
        .y(d => d.y))
      .style('fill', 'none')
      .style('stroke', '#CBD5E1')
      .style('stroke-width', 2);

    // 绘制节点组
    const nodeGroups = g.selectAll('.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', d => `translate(${d.x},${d.y})`)
      .style('cursor', 'pointer');

    // 节点背景矩形
    nodeGroups.append('rect')
      .attr('x', -80)
      .attr('y', -25)
      .attr('width', 160)
      .attr('height', 50)
      .attr('rx', 8)
      .style('fill', d => {
        const score = d.data.data.quality_score;
        if (score >= 0.9) return '#bbf7d0';  // 深绿色对应明显的绿背景 - Excellent
        if (score >= 0.7) return '#d1fae5';  // 绿色对应中绿背景 - Good
        if (score >= 0.5) return '#ecfdf5'; 
        return '#fee2e2';  // 浅红
      })
      .style('stroke', d => getQualityColor(d.data.data.quality_score))
      .style('stroke-width', 2);

    // 节点文本
    nodeGroups.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', -5)
      .style('font-size', '11px')
      .style('font-weight', 'bold')
      .style('fill', '#374151')
      .text(d => {
        const content = d.data.data.content;
        return content.length > 20 ? content.substring(0, 20) + '...' : content;
      });

    nodeGroups.append('text')
      .attr('x', 70)
      .attr('y', 0)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', 'bold')
      .style('fill', '#374151')
      .style('cursor', 'pointer')
      .text(d => {
        const originalNode = mockTreeData.beam_tree[d.data.id];
        if (originalNode.children && originalNode.children.length > 0) {
          return d.data.data._collapsed ? '+' : '−';
        }
        return '';
      });

    // 质量分数
    nodeGroups.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', 10)
      .style('font-size', '9px')
      .style('fill', '#6B7280')
      .text(d => `Score: ${d.data.data.quality_score.toFixed(2)}`);

    // 概率标签
    nodeGroups.append('text')
      .attr('text-anchor', 'middle') 
      .attr('dy', 20)
      .style('font-size', '8px')
      .style('fill', '#9CA3AF')
      .text(d => `P: ${d.data.data.probability.toFixed(2)}`);

    // 推理类型指示器
    nodeGroups.append('circle')
      .attr('cx', 70)
      .attr('cy', -20)
      .attr('r', 4)
      .style('fill', d => getReasoningColor(d.data.data.reasoning_type));

    // 交互事件
    nodeGroups.on('click', function(event, d) {
      setSelectedNode(d.data.data);
      nodeGroups.selectAll('rect').style('stroke-width', 2);
      d3.select(this).select('rect').style('stroke-width', 4);
    });
    nodeGroups.on('dblclick', function(event, d) {
    event.stopPropagation(); // 阻止事件冒泡
    
    const nodeId = d.data.id;
    const originalNode = mockTreeData.beam_tree[nodeId];
    
    // 只有有子节点的节点才能折叠/展开
    if (originalNode.children && originalNode.children.length > 0) {
      setCollapsedNodes(prev => {
        const newSet = new Set(prev);
        if (newSet.has(nodeId)) {
          newSet.delete(nodeId); // 展开
        } else {
          newSet.add(nodeId);    // 折叠
        }
        return newSet;
      });
    }
  });

    nodeGroups
    .on('mouseenter', function(event, d) {
      // 鼠标进入时的效果
      d3.select(this)
        .transition()
        .duration(200)
        .select('rect')
        .style('opacity', 0.9)
        .style('stroke-width', 3)
        .style('filter', 'drop-shadow(2px 2px 4px rgba(0,0,0,0.3))');
      
      // 显示详细信息的 tooltip
      const tooltip = d3.select('body')
        .append('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('background', 'rgba(0,0,0,0.8)')
        .style('color', 'white')
        .style('padding', '8px')
        .style('border-radius', '4px')
        .style('font-size', '12px')
        .style('pointer-events', 'none')
        .style('opacity', 0);
      
      tooltip.transition().duration(200).style('opacity', 1);
      tooltip.html(`
        <strong>${d.data.data.content}</strong><br/>
        Quality: ${d.data.data.quality_score.toFixed(3)}<br/>
        Probability: ${d.data.data.probability.toFixed(3)}<br/>
        Type: ${d.data.data.reasoning_type}<br/>
        Depth: ${d.data.data.depth}
      `);
    })
    .on('mousemove', function(event, d) {
      // 鼠标移动时更新 tooltip 位置
      d3.select('.tooltip')
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseleave', function(event, d) {
      // 鼠标离开时恢复原状
      d3.select(this)
        .transition()
        .duration(200)
        .select('rect')
        .style('opacity', 1)
        .style('stroke-width', 2)
        .style('filter', 'none');
      
      // 移除 tooltip
      d3.select('.tooltip').remove();
    });


    // 路径高亮功能
    const highlightPath = (pathId) => {
      const path = mockTreeData.paths.find(p => p.id === pathId);
      if (!path) return;

      // 重置样式
      links.style('stroke', '#CBD5E1').style('stroke-width', 2);
      nodeGroups.selectAll('rect').style('opacity', 0.3);

      // 高亮路径节点
      path.nodes.forEach(nodeId => {
        const node = root.descendants().find(d => d.data.id === nodeId);
        if (node) {
          const nodeGroup = d3.select(nodeGroups.nodes()[root.descendants().indexOf(node)]);
          nodeGroup.select('rect').style('opacity', 1);
        }
      });

      // 高亮路径连线
      for (let i = 1; i < path.nodes.length; i++) {
        const parentId = path.nodes[i - 1];
        const childId = path.nodes[i];
        const parentNode = root.descendants().find(d => d.data.id === parentId);
        const childNode = root.descendants().find(d => d.data.id === childId);
        
        if (parentNode && childNode) {
          links.filter(l => l.source === parentNode && l.target === childNode)
            .style('stroke', getQualityColor(path.score))
            .style('stroke-width', 4);
        }
      }
      
      setHighlightedPath(pathId);
    };

    // 重置高亮
    const resetHighlight = () => {
      links.style('stroke', '#CBD5E1').style('stroke-width', 2);
      nodeGroups.selectAll('rect').style('opacity', 1);
      setHighlightedPath(null);
    };

    // 导出函数到组件实例
    window.highlightPath = highlightPath;
    window.resetHighlight = resetHighlight;

  }, [collapsedNodes]);

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen">
      {/* 标题和问题 */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-4">
          Beam Search Reasoning
        </h1>
        
        {/* 问题框 */}
        <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-4">
          <h3 className="font-semibold text-blue-800 mb-2">Question</h3>
          <p className="text-blue-700 mb-3">{mockTreeData.problem.question}</p>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {mockTreeData.problem.options.map((option, idx) => (
              <div key={idx} className="text-blue-600 font-mono">{option}</div>
            ))}
          </div>
        </div>

        {/* 图例 */}
        <div className="flex flex-wrap gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className="font-medium">Quality:</span>
            <div className="flex gap-2">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-green-500"></div>
                <span>Excellent</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-blue-500"></div>
                <span>Good</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-yellow-500"></div>
                <span>Fair</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-red-500"></div>
                <span>Poor</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="font-medium">Reasoning Type:</span>
            <div className="flex gap-2">
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                <span>Understanding</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full bg-cyan-500"></div>
                <span>Calculation</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span>Conclusion</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* 主树形可视化 */}
        <div className="lg:col-span-3 bg-white rounded-lg shadow-lg p-4">
          <svg
            ref={svgRef}
            width="100%"
            height="800"
            viewBox="0 0 1200 800"
            className="border border-gray-200 rounded"
          ></svg>
        </div>

        {/* 右侧控制面板 */}
        <div className="bg-white rounded-lg shadow-lg p-4">
          <h3 className="text-lg font-semibold mb-4">Reasoning Paths</h3>
          
          {/* 路径控制 */}
          <div className="space-y-3 mb-6">
            <button
              onClick={() => window.resetHighlight()}
              className="w-full px-3 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 text-sm"
            >
              Reset View
            </button>
            
            {mockTreeData.paths.map((path, idx) => (
              <div key={path.id} className="space-y-2">
                <button
                  onClick={() => window.highlightPath(path.id)}
                  className={`w-full p-3 rounded-lg border-2 text-left transition-all ${
                    highlightedPath === path.id 
                      ? 'border-blue-500 bg-blue-50' 
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-sm">Path {idx + 1}</span>
                    <div className="flex items-center gap-2">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getQualityColor(path.score) }}
                      ></div>
                      <span className="text-xs">{path.score.toFixed(2)}</span>
                    </div>
                  </div>
                  <div className="text-xs text-gray-600 mb-1">
                    {path.nodes.length} steps
                  </div>
                  <div className="text-xs">
                    Answer: <span className="font-medium">{path.final_answer}</span>
                    {path.is_correct ? (
                      <span className="text-green-600 ml-1">✓</span>
                    ) : (
                      <span className="text-red-600 ml-1">✗</span>
                    )}
                  </div>
                </button>
              </div>
            ))}
          </div>

          {/* 选中节点详情 */}
          {selectedNode && (
            <div className="border-t pt-4">
              <h4 className="font-semibold mb-3">Multi-step Reasoning Chain</h4>
              <div className="bg-gray-50 rounded-lg p-3 mb-3">
                <div className="text-sm font-medium mb-2">{selectedNode.content}</div>
                <div className="space-y-1">
                  {selectedNode.variables.map((variable, idx) => (
                    <div key={idx} className="text-xs font-mono bg-white px-2 py-1 rounded border">
                      {variable}
                    </div>
                  ))}
                </div>
              </div>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between">
                  <span>Quality Score:</span>
                  <span className="font-medium">{selectedNode.quality_score.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Probability:</span>
                  <span className="font-medium">{selectedNode.probability.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Type:</span>
                  <span className="font-medium">{selectedNode.reasoning_type}</span>
                </div>
                <div className="flex justify-between">
                  <span>Depth:</span>
                  <span className="font-medium">{selectedNode.depth}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BeamSearchVisualizer;