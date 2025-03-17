use std::fmt;
use std::iter::FromIterator;

pub type Loc = usize;
const LOCSET_WORDS: usize = 2;
// u8::MAX = 256, so need 2 * u128 to cover the full range

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct Locset([u128; LOCSET_WORDS]);

impl Locset {
  pub fn new() -> Self {
    return Self([0u128; LOCSET_WORDS]);
  }

  pub fn iter<'a>(&'a self) -> impl Iterator<Item=Loc> + 'a {
    self.0.iter()
      .enumerate()
      .map(|(k, &bits)| Biterator::new(bits).map(move |i| i as Loc + ((k as Loc) << 7)))
      .flatten()
  }

  #[inline]
  pub fn insert(&mut self, i: Loc) {
    let (word_index, bit_index) = Self::word_bit_index(&i);
    self.0[word_index] |= 1 << bit_index;
  }

  #[inline]
  pub fn remove(&mut self, i: Loc) {
    let (word_index, bit_index) = Self::word_bit_index(&i);
    self.0[word_index] &= !(1 << bit_index);
  }

  #[inline]
  fn word_bit_index(i: &Loc) -> (usize, Loc) {
    let word_index = (i >> 7) as usize; // divide by 128 = 2**7, rounding down;
    let bit_index = i & 0x7f; // modulo 128
    return (word_index, bit_index);
  }

  #[inline]
  pub fn contains(&self, i: &Loc) -> bool {
    let (word_index, bit_index) = Self::word_bit_index(i);
    return (self.0[word_index] & (1 << bit_index)) != 0;
  }

  pub fn is_disjoint(&self, other: &Self) -> bool {
    self.0.iter().zip(other.0.iter()).all(|(x, y)| x & y == 0)
  }

  pub fn union(&self, other: &Self) -> Self {
    let mut new = self.clone();
    new.union_inplace(other);
    return new;
  }

  pub fn union_inplace(&mut self, other: &Self) {
    self.0.iter_mut().zip(other.0.iter()).for_each(|(x, y)| *x |= y);
  }

  pub fn intersect(&self, other: &Self) -> Self {
    let mut new = self.clone();
    new.intersect_inplace(other);
    return new;
  }

  pub fn xor(&self, other: &Self) -> Self {
    let mut new = self.clone();
    new.xor_inplace(other);
    return new;
  }

  pub fn intersect_inplace(&mut self, other: &Self) {
    self.0.iter_mut().zip(other.0.iter()).for_each(|(x, y)| *x &= y);
  }

  #[inline]
  pub fn subset(&self, other: &Self) -> bool {
    self.0.iter().zip(other.0.iter()).all(|(x, y)| *x & y == *x)
  }

  pub fn xor_inplace(&mut self, other: &Self) {
    self.0.iter_mut().zip(other.0.iter()).for_each(|(x, y)| *x ^= y);
  }

  pub fn len(&self) -> u128 {
    let mut count: u128 = 0;
    for _i in self.iter() {
      count += 1
    }
    count
  }
}


impl fmt::Debug for Locset {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_set()
      .entries(self.iter())
      .finish()
  }
}

impl<T: Into<u128>> FromIterator<T> for Locset {
  fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
    let mut words = [0u128; LOCSET_WORDS];
    for i in iter {
      let i: u128 = i.into();
      let word_index = (i >> 7) as usize; // divide by 128 = 2**7, rounding down;
      let bit_index = i & 0x7f; // modulo 128
      words[word_index] |= 1 << bit_index;
    }
    return Self(words);
  }
}


pub struct Biterator {
  bits : u128,
  ones : u32,
  next_index: u32,
}

impl Biterator {
  pub fn new(val : u128) -> Self {
    Self{ bits: val, ones: 0, next_index: 0 }
  }
}

impl Iterator for Biterator {
  type Item = u32;

  fn next(&mut self) -> Option<Self::Item> {
    if self.ones > 0 {
      let val = self.next_index;
      self.ones -= 1;
      self.next_index += 1;
      return Some(val);
    } else if self.bits == 0 {
      return None;
    } else {
      let nz = self.bits.trailing_zeros();
      self.bits >>= nz as u128;
      self.next_index += nz;
      let no = (!self.bits).trailing_zeros();
      self.ones = no;
      self.bits >>= no as u128;
      return self.next();
    }
  }
}