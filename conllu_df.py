#%%
from typing import Iterator,Callable, Any, Union, Optional, NewType
from conllu import TokenList, parse_incr, TokenTree
import pandas as pd

from itertools import chain, takewhile, count
import re

import utils
from utils import DataFrame

SENTENCE_ID = NewType('SENTENCE_ID', int)
MWE_ID = NewType('MWE_ID', str)
TOKEN_ID = NewType('TOKEN_ID', str)

# ----------------------------------
# redefining TokenTrees, the dirty way.
class TT(TokenTree):
	'''Custom hashable tokentree class

	Parameters
	----------
	TokenTree : TokenTree
		[description]
	'''
	def __hash__(self):
		try:
			return hash(frozenset(self.token.items()))
		except:
			print(self.token.items())
			return hash(frozenset(self.token.items()))
		# return hash(frozenset(self.token.items())) + \
		# 	hash(FrozenMultiSet(self.children))
	def __eq__(self, other):
		if not isinstance(other, self.__class__):
			return False
		return (self.token == other.token) and  \
			(utils.fmset(self.children) == utils.fmset(other.children)) #and \
			# (self.metadata == other.metadata)
	def __lt__(self, o):
		return hash(self) < hash(o)
	def __iter__(self):
		yield self
		for child in self.children:
			yield from child

def to_custom_TT(tokentree : TokenTree) -> TT:
	'''Converts a conllu.TokenTree into a custom TokenTree (TT).
	TTs have no dict or list in their token, and 
	redefine the __hash__, __eq__, __lt__, and __iter__ functions.

	'''
	return TT({
			k : v 
			for k,v in tokentree.token.items() 
			if not isinstance(v, dict) and not isinstance(v, list) 
		},
		[to_custom_TT(c) for c in tokentree.children],
		tokentree.metadata)

# ----------------------------------

class Conllu_df_parser:
	def __init__(
		self, 
		src : Union[str, list[TokenList]],
		preprocessing : Optional[list[Callable[[TokenList], TokenList]]] = None,
		postprocessing : Optional[list[Callable[[DataFrame], DataFrame]]] = None
	):
		'''Setup a parser which takes a .conll / .conllu / .cupt or and list of 
		TokenList and outputs a DataFrame where each feature is a column,
		each word a row, and TokenTree of each word is a the ...

		Parameters
		----------
		src : str | list[TokenList]
			either the path to the .conll / .conllu / .cupt file or list of Tokenlist.
		preprocessing : list[Callable[[TokenList], TokenList]] | None, optional
			list of functions (TokenList) -> TokenList to be executed before the
			Dataframe et TokenTree are generated, by default None
		postprocessing : list[Callable[[DataFrame], DataFrame]] | None, optional
			list of functions (DataFrame) -> DataFrame to be executed after the 
			DataFrame and the TokenTree are generated 
			(usefull when working with batches), by default None

		Raises
		------
		TypeError
			if src's type is incorrect
		'''
		if type(src) == str:
			file = open(src, 'r', encoding="utf-8")
			self.parser_TL = parse_incr(file)
		elif type(src) == list:
			self.parser_TL = src
		else :
			raise TypeError
			
		
		self._has_ended = False
		self._counter = 0
		self._preprocessing = preprocessing or [] # same as: preprocessing if preprocessing != None else []
		self._postprocessing = postprocessing or [] # it's magic

	def read_next_n(self, n : int = 0) -> Iterator[TokenList]:
		'''
		Iterates over the next n Tokenlists and apply the preprocessing
		'''
		for i, tl in enumerate(self.parser_TL):
			for f in self._preprocessing:
				tl = f(tl)
			yield tl
			if i >= n-1 and n != 0:
				return
		self._has_ended = True
	def get_df_per_batch(self, batch_size : int) -> Iterator[DataFrame]:
		def tt_to_tts(tree : TokenTree) -> list[TokenTree]:
			return [tree] + list(chain.from_iterable(tt_to_tts(child) for child in tree.children))

		def tl_to_ordered_tts(sentence : TokenList):
			return sorted(tt_to_tts(sentence.to_tree()), key = lambda x: x.token['id'])

		while not self._has_ended:
			df = DataFrame.from_dict(
				{
					(self._counter + m, token['id']): { 
						**token, 
						'TT' : next(tt) if type(token['id']) == int else None
					}
					for m, (tl, tt) in enumerate(map(
						lambda x : (
							x, 
							filter(lambda tt: tt.token['id'] != 0, tl_to_ordered_tts(x))
						),
						self.read_next_n(batch_size)
					))
					for token in tl
				},
				orient='index'
			)
			df.index = df.index.rename(['sentence_id', 'token_id'])
			for f in self._postprocessing:
				df = f(df)
			yield df
			self._counter += batch_size
	def get_df(self) -> DataFrame:
		return next(self.get_df_per_batch(0))

	def get_df_no_tt_per_batch(self, batch_size):
		df =  DataFrame.from_dict(
			{
				(self._counter + m, token['id']): {**token}
				for m, tl in enumerate(self.read_next_n(batch_size))
				for token in tl
			},
				orient='index'
		)
		df.index = df.index.rename(['sentence_id', 'token_id'])
		for f in self._postprocessing:
			df = f(df)
		yield df
		self._counter += batch_size
	def get_df_no_tt(self) -> DataFrame:
		return next(self.get_df_no_tt_per_batch(0))
	

def atomize(tl : TokenList) -> TokenList:
	'''
	atomize the column 'feats'
	'''
	for token in tl:
		token['feats'] = token['feats'] or {} # magic
		for k, v in token['feats'].items():
			token[k] = v
		del token['feats']
	return tl

def remove_compound(df : DataFrame) -> DataFrame:
	'''
	removes all rows of the df where 'id' is not an integer
	'''
	return df.loc[df['id'].apply(type) == int]
		

def locmap(df: DataFrame, column: str, func: Callable[[Any], bool]):
	'''function Monkey patched to pd.DataFrame(check on wiki)
	Given a dataframe, a column of said dataframe and a function
	returns the dataframe containing only those rows for which the function
	is True.
	
	Is kinda to `.loc` what `.applymap` is to `.apply`.
	`.loc` only takes function which can be vectorized. This method "fixes" that.
	'''
	return df.loc[lambda x: x[column].apply(func)]
DataFrame.locmap = locmap




def setup_data(
	file_path : str = 'dev.cupt',
	preproc_f : Optional[list] = None
) -> tuple[DataFrame[
	tuple[SENTENCE_ID], TT],
	DataFrame[tuple[SENTENCE_ID, TOKEN_ID], tuple
]]:
	def custom_TTs(df : DataFrame) -> DataFrame:
		df['TT'] = df['TT'].apply(to_custom_TT)
		return df

	preprocessing = [atomize] + (preproc_f if preproc_f else [])
	parser = Conllu_df_parser(
		file_path,
		preprocessing,
		[
			remove_compound, 
			custom_TTs
		]
	)
	df = parser.get_df()
	TTs = df.loc[df['head'] == 0][['TT']] \
		.droplevel(1) \
		.rename({'TT' : 'sentence'}, axis=1)
	df = df.drop('misc', axis=1)
	df = df.drop('deps', axis=1)
	return TTs, df

def setup_data_noTT(
	file_path : str = 'dev.cupt',
	preproc_f : Optional[list] = None,
	postproc_f : Optional[list] = (remove_compound,),
) -> tuple[DataFrame[
	tuple[SENTENCE_ID], TT],
	DataFrame[tuple[SENTENCE_ID, TOKEN_ID], tuple
]]:

	preprocessing = [atomize] + (preproc_f if preproc_f else [])
	parser = Conllu_df_parser(
		file_path,
		preprocessing,
		postproc_f

	)
	df = parser.get_df_no_tt()

	df = df.drop('misc', axis=1)
	df = df.drop('deps', axis=1)
	return df

def sort_column(x : DataFrame):
	res : dict[str, pd.Series]= {
		col : x[col].dropna().sort_values()
		for col in x.columns
	}
	res = {
		k : v
		for k, v in sorted(
			res.items(),
			key = lambda x : len(res[x[0]].unique()), reverse=True
		)
	}
	return res

def remove_NE(x):
	def f(i, x):
		return ';'.join([y for y in x.split(';') if re.search(f'^{i}(;|:|$)', y) == None]) or '*'
	def h(x):
		
		return [
			i
			for i, e, f in takewhile(
				lambda x: len(x[1]) != 0,
				map(
					lambda i : (
						i,
						x.filter(**{
							'parseme:mwe': lambda x: re.search(f'(;|^){i}(;|:|$)', x) != None
						}),
						x.filter(**{
							'parseme:mwe':
							lambda x: re.search(f'(?:;|^){i}:\w+?\|(NE).+?\|.+', x) != None
						})
					),
					count(1) 
				)
			)
			if len(f) != 0
		]

	res = x
	for i in h(x):
		for n in range(len(res)):
			res[n]['parseme:mwe'] = f(i, res[n]['parseme:mwe'])
	return res

def remove_NE(x):
	def f(i, x):
		return ';'.join([y for y in x.split(';') if re.search(f'^{i}(;|:|$)', y) == None]) or '*'
	def h(tl):
		
		return [
			i
			for i in {
				y 
				for x in tl
				for y in re.findall(f'(\d+)', x['parseme:mwe'])
			}
			if tl.filter(**{
				'parseme:mwe':
				lambda x: re.search(f'(?:;|^){i}:\w+?\|(NE).+?\|.+', x) != None
			})
		]

	res = x
	for i in h(x):
		for n in range(len(res)):
			res[n]['parseme:mwe'] = f(i, res[n]['parseme:mwe'])
	return res

def remove_VMWE(x):
	def f(i, x):
		return ';'.join([y for y in x.split(';') if re.search(f'^{i}(;|:|$)', y) == None]) or '*'
	def h(tl):
		
		return [
			i
			for i in {
				y 
				for x in tl
				for y in re.findall(f'(\d+)', x['parseme:mwe'])
			}
			if tl.filter(**{
				'parseme:mwe':
				lambda x: re.search(f'(?:;|^){i}:\w+?\|(MWE-).+?\|.+', x) != None
			})
		]

	res = x
	for i in h(x):
		for n in range(len(res)):
			res[n]['parseme:mwe'] = f(i, res[n]['parseme:mwe'])
	return res

def remove_nVMWE(x):
	def f(i, x):
		return ';'.join([y for y in x.split(';') if re.search(f'^{i}(;|:|$)', y) == None]) or '*'
	def h(tl):
		
		return [
			i
			for i in {
				y 
				for x in tl
				for y in re.findall(f'(\d+)', x['parseme:mwe'])
			}
			if tl.filter(**{
				'parseme:mwe':
				lambda x: re.search(f'(?:;|^){i}:\w+?\|(MWE(?!-)).*?\|.+', x) != None
			})
		]

	res = x
	for i in h(x):
		for n in range(len(res)):
			res[n]['parseme:mwe'] = f(i, res[n]['parseme:mwe'])
	return res


regex_map_cache = {i : re.compile(f'(;|^){i}(;|:|$)') for i in range(10)}
def regex_map(i):
	if not i in regex_map_cache:
		regex_map_cache[i] = re.compile(f'(;|^){i}(;|:|$)')
	return regex_map_cache[i]

regex1 = re.compile('(\d+)')



def get_mwes(df : DataFrame) -> DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple]:
	'''Returns a dataframe with (sentence_id, token_id, mwe_id) for index,
	with only token from MWEs. If a token is part of multiple MWE it is duplicated.

	'''
	try:
		return pd.concat([
			locmap(
				sentence,
				'parseme:mwe',
				lambda x: re.search(regex_map(i), x) != None
			).assign(mwe_id=i) # gets the token associated to the number i
			for _, sentence in df.groupby(level=0) # for each sentence
			for i in {
				y
				for x in sentence['parseme:mwe'].apply(
					lambda x: re.findall(regex1, x)
				)
				for y in x
			} # for each number associated to a MWE component in the sentence
		]).set_index('mwe_id', append=True)
	except:
		return locmap(
			df,
			'parseme:mwe',
			lambda x: re.search(f'(;|^){1}(;|:|$)', x) != None
		).assign(mwe_id=0).set_index('mwe_id', append=True)

	

def inline_mwes(
	mwes : DataFrame[tuple[SENTENCE_ID, TOKEN_ID, MWE_ID], tuple]
) -> DataFrame[tuple[SENTENCE_ID], tuple[utils.fmset, tuple]]:
	tmp = mwes.groupby(level=[0, 2]).apply(
		lambda x : (utils.fmset(list(x['lemma'])), tuple(x['id']))
	).apply(lambda x: pd.Series(x))
	return tmp.reset_index(
		'mwe_id',
		drop=True
	).reset_index().drop_duplicates().set_index('sentence_id')


