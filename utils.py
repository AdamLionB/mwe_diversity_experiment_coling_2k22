#%%
from __future__ import annotations

from typing import Iterator, TypeVar, Generic, Iterable
import pandas as pd

T = TypeVar('T')

A = TypeVar('A')
B = TypeVar('B')
class DataFrame(pd.DataFrame, Generic[A, B]):
	pass

def sorted_index_loc(serie, val, default_index, joker = ''):
	if val == joker or val is None :
		res=  default_index
	else:
		res = serie.iloc[
			serie.searchsorted(val, side='left'):
			serie.searchsorted(val, side='right')].index
	return res 


class fmset(Iterable, Generic[T]):
	'''Frozen (immutable) Multi-Set.

	no data is squished even when equal

	Parameters
	----------
	Iterable : Iterable
		Iterable of the data to put in the fmset
	Generic : Generic
		Generic type of data in the fmset

	Fields
	______
	data : dict[T, list[T]]
		Reference to equal objects are listed under the same key

	'''
	def __init__(self, l : Iterable[T]):
		'''[summary]

		Parameters
		----------
		l : Iterable[T]
			[description]
		'''
		self.data : dict[T, list[T]]= {}
		for k in l:
			if k not in self.data:
				self.data[k] = [k]
			else:
				self.data[k] += [k]
	def unholy_update(self, l):
		'''sorry'''
		self.data : dict[T, list[T]]= {}
		for k in l:
			if k not in self.data:
				self.data[k] = [k]
			else:
				self.data[k] += [k]
	def __iter__(self) -> Iterator[T]:
		try :
			tmp = list(sorted(self.data.items()))
		except : 
			tmp = list(self.data.items())
		for k, v in tmp:
			for x in v:
				yield x
	def __repr__(self) -> str:
		return str(tuple(self))
	def __str__(self) -> str:
		return str(tuple(self))
	def __len__(self) -> int:
		return sum(map(len,(self.data.values())))
	def __lt__(self, other) -> bool:
		return tuple(self) < tuple(other)
	def __eq__(self, other) -> bool:
		#FIXME comparing dict size might be more robust, or allow stanger uses
		return isinstance(other, fmset) and self.data == other.data
	def __hash__(self) -> int:
		return hash(tuple(self))
	def __contains__(self, o):
		if type(o) == fmset:
			for k, v in o.data.items():
				if k not in self.data:
					return False
				if len(self.data[k]) < len(o.data[k]):
					return False
			return True
		return o in self.data.keys()


	def __or__(self, o : fmset):
		#FIXME make the add return the actual references ? but what of the or/and
		# which references to return ?
		return fmset([
			k
			for k in self.data.keys() | o.data.keys()
			for _ in range(
				max(len(self.data.get(k, [])), len(o.data.get(k, [])))
			)
		])
	def __add__(self, o : fmset):
		return fmset([
			k
			for k in self.data.keys() & o.data.keys()
			for _ in range(len(self.data.get(k, []) + len(o.data.get(k, []))))
		])
	def __and__(self, other : fmset):
		return fmset([
			k
			for k in self.data.keys() & other.data.keys()
			for _ in range(min(len(self.data[k]), len(other.data[k])))
		])
	def jaccard(self, other):
		if not (s:=self | other):
			return 0
		return len(self & other) / len(s)
# %%
