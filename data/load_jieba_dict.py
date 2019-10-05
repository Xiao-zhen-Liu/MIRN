import re
ent_dict = set()
with open("new_zh_en_triple.txt", encoding="utf-8") as triples_reader:
	with open("zh_ent_dict.txt", "w", encoding="utf-8") as writer:
		for line in triples_reader.readlines():
			line = line.strip()
			triple_combination = re.findall(r"(?<=/)[^/]+(?=>)", line)
			ent_dict.add(triple_combination[0])
		for ent in ent_dict:
			writer.write(ent+"\n")