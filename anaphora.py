
import nltk 
from nltk.chunk import tree2conlltags 
from nltk.corpus import names 
import random 

class AnaphoraExample:
	def __init__(self):
		males = [(name,'male') for name in names.words('male.txt')]
		females = [(name,'female') for name in names.words('female.txt')]
		combined = males + females 
		random.shuffle(combined)
		training = [(self.feature(name),gender) for (name,gender) in combined]
		self._classifier = nltk.NaiveBayesClassifier.train(training)

	def feature(self,word):
		return {'last(1)':word[-1]}

	def gender(self,word):
		return self._classifier.classify(self.feature(word))

	def LearnAnaphora(self):
		sentences =[
		#Add your text here
		]

		for sent in sentences:
			chunk = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)),
				binary=False)

			stack = []
			print(sent)
			items = tree2conlltags(chunk)
			for item in items:
				if item[1] == 'NNP' and (item[2] == 'B-PERSON' or item[2] == 'O'):
					stack.append((item[0],self.gender(item[0])))
				elif item[1] == 'CC':
					stack.append(item[0])
				elif item[1] == 'PRP':
					stack.append(item[0])

			print("\t {}".format(stack))

anaphora = AnaphoraExample()
anaphora.LearnAnaphora()
