#%%
from __future__ import annotations
from abc import abstractmethod, ABCMeta
from typing import TypeVar, Generic, Iterable, Type, Callable, Any, Optional 
from collections import defaultdict
from itertools import product
from math import log


import pandas as pd
from pandas import DataFrame

from scipy.optimize import curve_fit
from scipy.stats import zipfian



import conllu_df
from conllu_df import SENTENCE_ID, TOKEN_ID, MWE_ID

import utils
from utils import sorted_index_loc

T = TypeVar('T')
U = TypeVar('U')

class MWE_lexicon(Generic[T]):
	def __init__(self, content: Iterable[T]):
		self.content = content
	def __iter__(self):
		yield from self.content
	def __str__(self) -> str:
		return str(self.content)
	def __repr__(self) -> str:
		return str(self)

	@property
	@abstractmethod #
	def T_cl(self) -> Type[T]: ...
	T_cl: Type[T]

	@abstractmethod
	def match(self): ...
	@staticmethod
	def concretize(
		t_cl : Type[T],
		match_method: Callable[[MWE_lexicon[T], Any], Any]
	) -> Type[MWE_lexicon[T]]:
		class X(MWE_lexicon):
			match = match_method
			T_cl = t_cl
		return X

class Lexicon_formalism(Generic[T], metaclass=ABCMeta):
	
	@property
	@abstractmethod #
	def lexicon_cl(self) -> Type[MWE_lexicon[T]]: ...
	lexicon_cl : Type[MWE_lexicon[T]] #tricks to get the right type inference

	@abstractmethod
	def instantiate(self, data) -> MWE_lexicon[T]: ...
	@staticmethod
	def concretize(
		t_cl : Type[T],
		instantiatiation_method: Callable[[Any], MWE_lexicon[T]],
		match_method: Callable[[MWE_lexicon[T], Any], Any]
	)-> Lexicon_formalism[T]:
		class X(Lexicon_formalism):
			instantiate = instantiatiation_method
			lexicon_cl = MWE_lexicon.concretize(t_cl, match_method)
		return X()

class Seq_rep(Generic[T, U]):

	def __init__(self, component, insertions):
		self.components = component
		self.insertions = insertions

	def __eq__(self, o: object) -> bool:
		if type(o) == type(self):
			return self.components == o.components and self.insertions == o.insertions
		return False
	
	def __hash__(self) -> int:
		return (
			hash(tuple([frozenset(x.items()) for x in self.components]))
			+hash(tuple(self.insertions))
		)

	def __str__(self) -> str:
		return str(self.components)+'\ '+str(self.insertions)
	
	def __repr__(self) -> str:
		return str(self)

	@classmethod
	def handle_discontinuities(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
		return '*'

	@property
	@abstractmethod #
	def properties(self) -> T: ...
	properties : T

	@staticmethod
	def concretize(prop: T, discontinuity_handler: U = None) -> Type[Seq_rep[T, U]]:
		class X(Seq_rep):
			properties = prop
			if discontinuity_handler != None:
				handle_discontinuities = discontinuity_handler
		return X
	@classmethod
	def from_mwe(
		cl,
		sentence: DataFrame[tuple[SENTENCE_ID, TOKEN_ID], tuple],
		mwe: DataFrame
	):
		component = []
		insertions = []
		last_id = mwe.iloc[0]['id']
		for _, row in mwe.iterrows():
			if row['id'] - last_id >= 1:
				try:
					insertions.append(
						cl.handle_discontinuities(sentence.loc[last_id+1:row['id']-1])
					)
				except InvalidInsertion:
					return 
			component.append({prop: row[prop] for prop in cl.properties})
			last_id = row['id']

		# print('---')
		return cl(component, insertions)
	def match(self, df, sid):
		ress = [[]]*len(self.components)
		for n, component in enumerate(self.components):
			for k,v in component.items():
				if not ress[n] :
					ress[n] = set(sorted_index_loc(sid[k], v, sid['id'].index))
				else:
					ress[n] = ress[n].intersection(
						set(sorted_index_loc(sid[k], v, sid['id'].index))
					)

		matches = [
			[
				[v.name for k, v in b.iterrows()]
				for _, b in df.loc[list(a)].groupby(level=0)
			] for a in ress
		]
		sentences= [defaultdict(list) for _ in range(len(self.components))]
		for n, x in enumerate(matches):
			for y in x:
				sentences[n][y[0][0]]+=y
		
		res=[]
		keys = set.intersection(*[set(x.keys()) for x in sentences])
		# print(sentences)
		for k in keys:
			for prod in product(*[x[k] for x in sentences]):
				# print(prod)
				if [e[1] for e in prod] != sorted([e[1] for e in prod]):
					# print([e[1] for e in prod])
					continue

				prod = [df.loc[e] for e in prod]
				# print(prod)				
				for insertion, (a, b) in zip(self.insertions, zip(prod[:-1],prod[1:])):
					if b['id'] - a['id'] > 1:
						# print(df.loc[a.name[0]].loc[a['id']+1:b['id']-1])
						try:
							tmp = self.__class__.handle_discontinuities(df.loc[a.name[0]].loc[a['id']+1:b['id']-1])
							if insertion != tmp:
								break
						except InvalidInsertion:
							break
				else:
					res.append(pd.DataFrame(prod))
		# print(res)
		return res	

	def match2(self, sentence):
		matches = [	
			sentence.loc[
				lambda x: x.apply(
					lambda y : all(
						y[k] == v for k,v in component.items()
					), axis=1
				)
			] for component in self.components
		]
		res = []
		for prod in product(*[[v for k, v in x.iterrows()] for x in matches]):
			if [e['id'] for e in prod] != sorted([e['id'] for e in prod]):
				break
			
			for insertion, (a, b) in zip(self.insertions, zip(prod[:-1],prod[1:])):
				if b['id'] - a['id'] > 1:
					tmp = self.__class__.handle_discontinuities(sentence.loc[a['id']+1:b['id']-1])
					if insertion != tmp:
						break
			else:
				res.append(pd.DataFrame(prod))
		return res


# ---------- Seq_rep lexicon functions --------
# why some of those are not class methods ? who knows

def extract_pattern_from_data(
	lf: Lexicon_formalism[Seq_rep],
	df: DataFrame[tuple[SENTENCE_ID, TOKEN_ID], tuple]
) -> MWE_lexicon[Seq_rep]:
	return lf.lexicon_cl({
		tmp
		for _, sentence in df.groupby(level=0)
		if len(mwes:=conllu_df.get_mwes(sentence)) != 0
		for _, mwe in mwes.groupby(level=2)
		if (tmp:=lf.lexicon_cl.T_cl.from_mwe(sentence.droplevel(0), mwe))
	})

def SeqRep_match(
	self : MWE_lexicon[Seq_rep],
	df: DataFrame[tuple[SENTENCE_ID, TOKEN_ID], tuple],
	sid
):
	tmp = [
		e
		for pattern in self
		# for _, sentence in df.groupby(level=0)
		for e in pattern.match(df, sid)
	]
	# print(tmp)
	tmp = pd.concat(
		[x.assign(mwe_id=n) for n, x in enumerate(tmp)]
	).set_index('mwe_id', append=True)

	tmp.index= tmp.index.set_names(['sentence_id', 'token_id'], level=[0, 1])
	return tmp

class InvalidInsertion(Exception): ...

@classmethod
def disc0(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'contiguous'
	if len(insertions) > 0:
		raise InvalidInsertion

@classmethod
def disc_lemma1(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''1 pos'''
	if len(insertions) > 1:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

@classmethod
def disc_lemma2(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''1 pos'''
	if len(insertions) > 2:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

@classmethod
def disc_lemma3(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''1 pos'''
	if len(insertions) > 3:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

@classmethod
def disc_lemma4(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''1 pos'''
	if len(insertions) > 4:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

@classmethod
def disc_lemma5(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''1 pos'''
	if len(insertions) > 5:
		raise InvalidInsertion
	return tuple(insertions['lemma'])

@classmethod
def disc_lemma0(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''list lemma'''
	return tuple(insertions['lemma'])



@classmethod
def disc_pos1(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''1 pos'''
	if len(insertions) > 1:
		raise InvalidInsertion
	return tuple(insertions['upos'])

@classmethod
def disc_pos2(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	''' 2 pos max'''
	if len(insertions) > 2:
		raise InvalidInsertion
	return tuple(insertions['upos'])

@classmethod
def disc_pos3(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	''' 3 pos max'''
	if len(insertions) > 3:
		raise InvalidInsertion
	return tuple(insertions['upos'])

@classmethod
def disc_pos4(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	''' 4 pos max'''
	if len(insertions) > 4:
		raise InvalidInsertion
	return tuple(insertions['upos'])

@classmethod
def disc_pos5(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	''' 5 pos max'''
	if len(insertions) > 5:
		raise InvalidInsertion
	return tuple(insertions['upos'])

@classmethod
def disc_pos0(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''list pos'''
	return tuple(insertions['upos'])


# @classmethod
# def disc2(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
# 	'''+ or "" '''
# 	if len(insertions) >= 1:
# 		return '+'
# 	return ''

@classmethod
def disc_1(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 1 '''
	if len(insertions) > 1:
		raise InvalidInsertion
	return '*'

@classmethod
def disc_2(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 2 '''
	if len(insertions) > 2:
		raise InvalidInsertion
	return '*'

@classmethod
def disc_3(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 3 '''
	if len(insertions) > 3:
		raise InvalidInsertion
	return '*'


@classmethod
def disc_4(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 4'''
	if len(insertions) > 4:
		raise InvalidInsertion
	return '*'

@classmethod
def disc_5(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''* size 5 '''
	if len(insertions) > 5:
		raise InvalidInsertion
	return '*'

@classmethod
def disc_0(cl, insertions: DataFrame[tuple[TOKEN_ID], tuple]):
	'''*'''
	return '*'

disc_fs = [disc0, 
			disc_lemma1, disc_lemma2, disc_lemma3, disc_lemma4, disc_lemma5, disc_lemma0,
			disc_pos1, disc_pos2, disc_pos3, disc_pos4, disc_pos5, disc_pos0,
			disc_1, disc_2, disc_3, disc_4, disc_5, disc_0]



#  ------------------ other

def get_tp(truth, pred):
	return pd.merge(
		conllu_df.inline_mwes(truth).reset_index(),
		conllu_df.inline_mwes(pred).reset_index(),
		how='inner'
	)




def evaluate(
	truth: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID]],
	pred: DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID]],
	true_pred : Optional[DataFrame] = None
):
	inline_truth = conllu_df.inline_mwes(truth)
	inline_pred = conllu_df.inline_mwes(pred)
	if true_pred is None :
		true_pred = pd.merge(
			inline_truth .reset_index(),
			inline_pred.reset_index()
			, how='inner'
		)
	p = len(true_pred)/len(inline_pred)
	r = len(true_pred)/len(inline_truth)
	return {
		'p': p,
		'r': r,
		'f': (2 * p * r) / (p + r)
	} | diversity_eval(true_pred)


def diversity_eval(true_pred: DataFrame[tuple[SENTENCE_ID], tuple[utils.fmset, tuple]]):
	tp_grpby = true_pred.assign(n=1).groupby(0).count()['n']
	n = sum(tp_grpby)
	tp_grpby = tp_grpby / sum(tp_grpby)

	s = curve_fit(
		lambda x, s: zipfian.pmf(x, s, len(tp_grpby)),
		list(range(1, len(tp_grpby)+1)),
		tp_grpby.sort_values(ascending=False)
	)[0][0]

	return {
		'richness': len(tp_grpby),
		'normalize_r': len(tp_grpby)/n,
		'e10': E(1, 0, tp_grpby),
		'e21': E(2, 1, tp_grpby),
		'1/S': 1/s
	}


def Na(a, p):
	if a == 1:
		return 2 ** (- sum(x * log(x, 2) if x != 0 else 0 for x in p))
	try :
		return sum(x ** a for x in p) ** (1 / (1 - a))
	except ZeroDivisionError:
		return 0

def E(n, m, p):
	return (Na(n, p) / Na(m, p)) if Na(m, p) else 0

