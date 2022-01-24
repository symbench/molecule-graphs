import xml.etree.ElementTree as ET
import sys
file_name = "all_schema_UAV.graphml"
node_id_seed = 105312256
tree = ET.parse(file_name)
root = tree.getroot()
graph = None
output_filename = "quadcopter.graphml"
output = open(output_filename, 'w')

for child in root:
	for child2 in child:
		if child2.tag[-4:]=='node':
			if int(child2.attrib['id']) == node_id_seed:
				graph = child

elements = []

target_ids = [node_id_seed]

while len(target_ids) > 0:

	for target_id in target_ids:
		print("nodes in queue: {}".format(len(target_ids)))
		for child in graph:
			if child.tag[-4:]=='node':
				if int(child.attrib['id']) == target_id:
					elements.append(child)
					graph.remove(child)
			if child.tag[-4:]=='edge':
				if int(child.attrib['target']) == target_id:
					elements.append(child)
					target_ids.append(int(child.attrib['source']))
					graph.remove(child)
		target_ids.remove(target_id)


string_builder = """<?xml version='1.0' encoding='UTF-8'?>\n
						<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.1/graphml.xsd">\n
							<key id="[]Definition" for="node" attr.name="[]Definition" attr.type="string"/>\n
							<key id="[]Name" for="node" attr.name="[]Name" attr.type="string"/>\n
							<key id="[http://www.w3.org/2001/XMLSchema-instance]type" for="node" attr.name="[http://www.w3.org/2001/XMLSchema-instance]type" attr.type="string"/>\n
							<key id="[]SurfaceReverseMap" for="node" attr.name="[]SurfaceReverseMap" attr.type="string"/>\n
							<key id="[]Unit" for="node" attr.name="[]Unit" attr.type="string"/>\n
							<key id="[]Format" for="node" attr.name="[]Format" attr.type="string"/>\n
							<key id="[]DimensionType" for="node" attr.name="[]DimensionType" attr.type="string"/>\n
							<key id="[]Author" for="node" attr.name="[]Author" attr.type="string"/>\n
							<key id="[]DatumName" for="node" attr.name="[]DatumName" attr.type="string"/>\n
							<key id="_partitionV" for="node" attr.name="_partition" attr.type="string"/>\n
							<key id="[]Locator" for="node" attr.name="[]Locator" attr.type="string"/>\n
							<key id="[]CName" for="node" attr.name="[]CName" attr.type="string"/>\n
							<key id="[]XPosition" for="node" attr.name="[]XPosition" attr.type="string"/>\n
							<key id="[]ID" for="node" attr.name="[]ID" attr.type="string"/>\n
							<key id="[]Version" for="node" attr.name="[]Version" attr.type="string"/>\n
							<key id="[]Hash" for="node" attr.name="[]Hash" attr.type="string"/>\n
							<key id="[]Class" for="node" attr.name="[]Class" attr.type="string"/>\n
							<key id="value" for="node" attr.name="value" attr.type="string"/>\n
							<key id="[]Path" for="node" attr.name="[]Path" attr.type="string"/>\n
							<key id="[]OnDataSheet" for="node" attr.name="[]OnDataSheet" attr.type="string"/>\n
							<key id="[]YPosition" for="node" attr.name="[]YPosition" attr.type="string"/>\n
							<key id="[]Dimensions" for="node" attr.name="[]Dimensions" attr.type="string"/>\n
							<key id="[]SchemaVersion" for="node" attr.name="[]SchemaVersion" attr.type="string"/>\n
							<key id="[]DataType" for="node" attr.name="[]DataType" attr.type="string"/>\n
							<key id="labelV" for="node" attr.name="labelV" attr.type="string"/>\n
							<key id="[]Notes" for="node" attr.name="[]Notes" attr.type="string"/>\n
							<key id="statusV" for="node" attr.name="status" attr.type="string"/>\n
							<key id="_partitionE" for="edge" attr.name="_partition" attr.type="string"/>\n
							<key id="labelE" for="edge" attr.name="labelE" attr.type="string"/>\n
							<key id="statusE" for="edge" attr.name="status" attr.type="string"/>\n
							<graph id="G" edgedefault="directed">\n"""
for element in elements:
	conv_string = ET.tostring(element).decode('utf-8').replace('xmlns:ns0="http://graphml.graphdrawing.org/xmlns"', '').replace('ns0:', '')
	string_builder += conv_string + "\n"
string_builder += "</graph></graphml>"
output.write(string_builder)


