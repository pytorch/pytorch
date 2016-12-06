class A < B; def self.create(object = User) object end end
class Zebra; def inspect; "X#{2 + self.object_id}" end end

module ABC::DEF
  include Comparable

  # @param test
  # @return [String] nothing
  def foo(test)
    Thread.new do |blockvar|
      ABC::DEF.reverse(:a_symbol, :'a symbol', :<=>, 'test' + ?\012)
      answer = valid?4 && valid?CONST && ?A && ?A.ord
    end.join
  end

  def [](index) self[index] end
  def ==(other) other == self end
end

class Car < ActiveRecord::Base
  has_many :wheels, class_name: 'Wheel', foreign_key: 'car_id'
  scope :available, -> { where(available: true) }
end

hash = {1 => 'one', 2 => 'two'}

2.0.0p0 :001 > ['some']
 => ["some"]
