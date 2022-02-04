# clean data from xml file

import xml.dom.minidom

dom = xml.dom.minidom.parse('laptops_data/raw_data/Laptops_Train.xml')
root = dom.documentElement

sentences = root.getElementsByTagName('sentence')

instances = []
elmo_text = []

for sentence in sentences:
    text = sentence.getElementsByTagName('text')[0].firstChild.data
    aspectTerms = sentence.getElementsByTagName('aspectTerm')
    for aspectTerm in aspectTerms:
        begin_index = int(aspectTerm.getAttribute('from'))
        end_index = int(aspectTerm.getAttribute('to'))
        if text[begin_index:end_index].strip() == aspectTerm.getAttribute('term').strip():
            instance = ((text[:begin_index] + ' ' + text[end_index:]).strip(), aspectTerm.getAttribute('term'), aspectTerm.getAttribute('polarity'))
            instances.append(instance)
        else:
            print text
            print text[begin_index:end_index].strip()
            print aspectTerm.getAttribute('term').strip()
        if aspectTerm.getAttribute('polarity') in ['positive', 'neutral', 'negative']:
            elmo_text.append((text[:begin_index] + ' ' + aspectTerm.getAttribute('term') + ' ' + text[end_index:]).strip())



text = open('laptops_data/processed_data/train.text', 'w')
aspect = open('laptops_data/processed_data/train.aspect', 'w')
polarity = open('laptops_data/processed_data/train.polarity', 'w')

for instance in instances:
    if instance[2] == 'positive':
        text.write(instance[0] + '\n')
        aspect.write(instance[1] + '\n')
        polarity.write('0\n')
    elif instance[2] == 'neutral':
        text.write(instance[0] + '\n')
        aspect.write(instance[1] + '\n')
        polarity.write('1\n')
    elif instance[2] == 'negative':
        text.write(instance[0] + '\n')
        aspect.write(instance[1] + '\n')
        polarity.write('2\n')

text.close()
aspect.close()
polarity.close()

dom = xml.dom.minidom.parse('laptops_data/raw_data/Laptops_Test_Gold.xml')
root = dom.documentElement

sentences = root.getElementsByTagName('sentence')

instances = []

for sentence in sentences:
    text = sentence.getElementsByTagName('text')[0].firstChild.data
    aspectTerms = sentence.getElementsByTagName('aspectTerm')
    for aspectTerm in aspectTerms:
        begin_index = int(aspectTerm.getAttribute('from'))
        end_index = int(aspectTerm.getAttribute('to'))
        if text[begin_index:end_index].strip() == aspectTerm.getAttribute('term').strip():
            instance = ((text[:begin_index] + ' ' + text[end_index:]).strip(), aspectTerm.getAttribute('term'), aspectTerm.getAttribute('polarity'))
            instances.append(instance)
        else:
            print text
            print text[begin_index:end_index].strip()
            print aspectTerm.getAttribute('term').strip()
        if aspectTerm.getAttribute('polarity') in ['positive', 'neutral', 'negative']:
            elmo_text.append((text[:begin_index] + ' ' + aspectTerm.getAttribute('term') + ' ' + text[end_index:]).strip())


text = open('laptops_data/processed_data/test.text', 'w')
aspect = open('laptops_data/processed_data/test.aspect', 'w')
polarity = open('laptops_data/processed_data/test.polarity', 'w')

for instance in instances:
    if instance[2] == 'positive':
        text.write(instance[0] + '\n')
        aspect.write(instance[1] + '\n')
        polarity.write('0\n')
    elif instance[2] == 'neutral':
        text.write(instance[0] + '\n')
        aspect.write(instance[1] + '\n')
        polarity.write('1\n')
    elif instance[2] == 'negative':
        text.write(instance[0] + '\n')
        aspect.write(instance[1] + '\n')
        polarity.write('2\n')

text.close()
aspect.close()
polarity.close()

print len(elmo_text)
elmo_text = set(elmo_text)
print len(elmo_text)
elmo = open('laptops_data/elmo.train', 'w')
for text in elmo_text:
    elmo.write(text + '\n')
elmo.close()

