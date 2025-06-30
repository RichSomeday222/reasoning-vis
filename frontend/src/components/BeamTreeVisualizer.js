// src/components/BeamTreeVisualizer.js
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const BeamTreeVisualizer = ({ 
  displayData, 
  collapsedNodes, 
  setCollapsedNodes, 
  setSelectedNode, 
  setHighlightedPath 
}) => {
  const svgRef = useRef();

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
    if (!svgRef.current || !displayData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 1200;
    const height = 800;
    const margin = {top: 50, right: 50, bottom: 50, left: 50};

    const buildHierarchy = () => {
      const nodes = displayData.beam_tree;
      
      const buildNode = (nodeId) => {
        const node = nodes[nodeId];
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
        if (score >= 0.9) return '#bbf7d0';
        if (score >= 0.7) return '#d1fae5';
        if (score >= 0.5) return '#ecfdf5'; 
        return '#fee2e2';
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

    // 折叠/展开按钮
    nodeGroups.append('text')
      .attr('x', 70)
      .attr('y', 0)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', 'bold')
      .style('fill', '#374151')
      .style('cursor', 'pointer')
      .text(d => {
        const originalNode = displayData.beam_tree[d.data.id];
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
      event.stopPropagation();
      
      const nodeId = d.data.id;
      const originalNode = displayData.beam_tree[nodeId];
      
      if (originalNode.children && originalNode.children.length > 0) {
        setCollapsedNodes(prev => {
          const newSet = new Set(prev);
          if (newSet.has(nodeId)) {
            newSet.delete(nodeId);
          } else {
            newSet.add(nodeId);
          }
          return newSet;
        });
      }
    });

    // 路径高亮功能
    const highlightPath = (pathId) => {
      const path = displayData.paths.find(p => p.id === pathId);
      if (!path) return;

      links.style('stroke', '#CBD5E1').style('stroke-width', 2);
      nodeGroups.selectAll('rect').style('opacity', 0.3);

      path.nodes.forEach(nodeId => {
        const node = root.descendants().find(d => d.data.id === nodeId);
        if (node) {
          const nodeGroup = d3.select(nodeGroups.nodes()[root.descendants().indexOf(node)]);
          nodeGroup.select('rect').style('opacity', 1);
        }
      });

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

    const resetHighlight = () => {
      links.style('stroke', '#CBD5E1').style('stroke-width', 2);
      nodeGroups.selectAll('rect').style('opacity', 1);
      setHighlightedPath(null);
    };

    // 导出函数到全局
    window.highlightPath = highlightPath;
    window.resetHighlight = resetHighlight;

  }, [collapsedNodes, displayData, setCollapsedNodes, setSelectedNode, setHighlightedPath]);

  return (
    <div className="bg-white rounded-lg shadow-lg p-4">
      <svg
        ref={svgRef}
        width="100%"
        height="800"
        viewBox="0 0 1200 800"
        className="border border-gray-200 rounded"
      ></svg>
    </div>
  );
};

export default BeamTreeVisualizer;