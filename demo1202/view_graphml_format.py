#!/usr/bin/env python3
"""
查看 GraphML 文件的数据格式
"""
import xml.etree.ElementTree as ET
import sys
import os
from collections import Counter

def analyze_graphml(file_path, max_nodes=5, max_edges=5):
    """
    分析 GraphML 文件的数据格式
    
    Args:
        file_path: GraphML 文件路径
        max_nodes: 最多显示的节点数
        max_edges: 最多显示的边数
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"文件路径: {file_path}")
    print(f"文件大小: {file_size / (1024*1024):.2f} MB")
    print("=" * 80)
    
    try:
        # 解析 XML
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # 定义命名空间
        namespace = {'': 'http://graphml.graphdrawing.org/xmlns'}
        
        # 打印根元素信息
        print(f"根元素: {root.tag}")
        print(f"根元素属性: {root.attrib}")
        print()
        
        # 查找所有 key 定义
        print("=" * 80)
        print("Key 定义（数据字段）:")
        print("=" * 80)
        keys = root.findall('.//key', namespace)
        
        node_keys = {}
        edge_keys = {}
        for key in keys:
            key_id = key.get('id')
            key_name = key.get('attr.name')
            key_type = key.get('attr.type')
            key_for = key.get('for')
            
            if key_for == 'node':
                node_keys[key_id] = {'name': key_name, 'type': key_type}
            elif key_for == 'edge':
                edge_keys[key_id] = {'name': key_name, 'type': key_type}
            
            print(f"  {key_id}: {key_name} ({key_type}) - for {key_for}")
        print()
        
        # 统计节点和边
        nodes = root.findall('.//node', namespace)
        edges = root.findall('.//edge', namespace)
        
        print("=" * 80)
        print("数据统计:")
        print("=" * 80)
        print(f"节点数量: {len(nodes)}")
        print(f"边数量: {len(edges)}")
        print()
        
        # 统计 entity_type 分布
        if len(nodes) > 0:
            entity_types = []
            for node in nodes:
                entity_type_elem = node.find("./data[@key='d1']", namespace)
                if entity_type_elem is not None and entity_type_elem.text:
                    entity_types.append(entity_type_elem.text.strip('"'))
            
            type_counter = Counter(entity_types)
            print("=" * 80)
            print("Entity Type 分布 (Top 10):")
            print("=" * 80)
            for entity_type, count in type_counter.most_common(10):
                print(f"  {entity_type}: {count}")
            print()
        
        # 查看示例节点的完整结构
        if len(nodes) > 0:
            print("=" * 80)
            print(f"前 {min(max_nodes, len(nodes))} 个节点的完整结构:")
            print("=" * 80)
            for i, node in enumerate(nodes[:max_nodes]):
                print(f"\n节点 {i+1}:")
                print(f"  节点ID: {node.get('id')}")
                print("  节点属性:")
                for data in node.findall('./data', namespace):
                    key_id = data.get('key')
                    key_info = node_keys.get(key_id, {})
                    key_name = key_info.get('name', key_id)
                    value = data.text if data.text else ''
                    # 限制显示长度
                    if len(value) > 150:
                        display_value = value[:150] + '...'
                    else:
                        display_value = value
                    print(f"    {key_name} ({key_id}): {display_value}")
            print()
        
        # 查看示例边的完整结构
        if len(edges) > 0:
            print("=" * 80)
            print(f"前 {min(max_edges, len(edges))} 个边的完整结构:")
            print("=" * 80)
            for i, edge in enumerate(edges[:max_edges]):
                print(f"\n边 {i+1}:")
                print(f"  源节点: {edge.get('source')}")
                print(f"  目标节点: {edge.get('target')}")
                print("  边属性:")
                for data in edge.findall('./data', namespace):
                    key_id = data.get('key')
                    key_info = edge_keys.get(key_id, {})
                    key_name = key_info.get('name', key_id)
                    value = data.text if data.text else ''
                    # 限制显示长度
                    if len(value) > 150:
                        display_value = value[:150] + '...'
                    else:
                        display_value = value
                    print(f"    {key_name} ({key_id}): {display_value}")
            print()
        
        # 数据格式总结
        print("=" * 80)
        print("数据格式总结:")
        print("=" * 80)
        print("\n节点 (Node) 字段:")
        for key_id in sorted(node_keys.keys()):
            key_info = node_keys[key_id]
            print(f"  - {key_info['name']} ({key_id}): {key_info['type']}")
        
        print("\n边 (Edge) 字段:")
        for key_id in sorted(edge_keys.keys()):
            key_info = edge_keys[key_id]
            print(f"  - {key_info['name']} ({key_id}): {key_info['type']}")
        
        print("\n" + "=" * 80)
        print("GraphML 文件结构:")
        print("=" * 80)
        print("""
<graphml>
  <key id="d0" for="node" attr.name="entity_id" attr.type="string"/>
  <key id="d1" for="node" attr.name="entity_type" attr.type="string"/>
  <key id="d2" for="node" attr.name="description" attr.type="string"/>
  <key id="d3" for="node" attr.name="source_id" attr.type="string"/>
  <key id="d4" for="node" attr.name="file_path" attr.type="string"/>
  <key id="d5" for="node" attr.name="created_at" attr.type="long"/>
  
  <key id="d6" for="edge" attr.name="weight" attr.type="double"/>
  <key id="d7" for="edge" attr.name="description" attr.type="string"/>
  <key id="d8" for="edge" attr.name="keywords" attr.type="string"/>
  <key id="d9" for="edge" attr.name="source_id" attr.type="string"/>
  <key id="d10" for="edge" attr.name="file_path" attr.type="string"/>
  <key id="d11" for="edge" attr.name="created_at" attr.type="long"/>
  
  <graph>
    <node id="实体名称">
      <data key="d0">实体ID</data>
      <data key="d1">实体类型</data>
      <data key="d2">描述</data>
      <data key="d3">来源chunk ID</data>
      <data key="d4">文件路径</data>
      <data key="d5">创建时间戳</data>
    </node>
    
    <edge source="源实体" target="目标实体">
      <data key="d6">权重</data>
      <data key="d7">关系描述</data>
      <data key="d8">关系关键词</data>
      <data key="d9">来源chunk ID</data>
      <data key="d10">文件路径</data>
      <data key="d11">创建时间戳</data>
    </edge>
  </graph>
</graphml>
        """)
        
    except ET.ParseError as e:
        print(f"XML解析错误: {e}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    # 默认查看 metaqa_demo 的 graphml
    default_file = "/home/infres/zzhu-24/kg-rag/LightRAG/cache/metaqa_demo/graph_chunk_entity_relation.graphml"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
    
    max_nodes = 5
    max_edges = 5
    if len(sys.argv) > 2:
        max_nodes = int(sys.argv[2])
    if len(sys.argv) > 3:
        max_edges = int(sys.argv[3])
    
    analyze_graphml(file_path, max_nodes, max_edges)


if __name__ == "__main__":
    main()

