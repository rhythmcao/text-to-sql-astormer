#coding=utf-8
import re, os
from itertools import chain
from collections import OrderedDict


def remove_comment(text):
    text = re.sub(re.compile("#.*"), "", text)
    text = '\n'.join(filter(lambda x: x, text.split('\n')))
    return text


class ASDLGrammar(object):
    """ Collection of types, fields, constructors and productions
    """
    def __init__(self, productions, file_path):
        # productions are indexed by their head types
        self._grammar_name = os.path.splitext(os.path.basename(file_path))[0]
        self._productions = OrderedDict()
        self._constructor_production_map = dict()
        for prod in productions:
            if prod.type not in self._productions:
                self._productions[prod.type] = list()
            self._productions[prod.type].append(prod)
            if prod.constructor.name in self._constructor_production_map:
                raise ValueError('[ERROR]: duplicate constructor name %s' % (prod.constructor.name))
            self._constructor_production_map[prod.constructor.name] = prod

        # root type is the first defined non-terminal
        self.root_type = productions[0].type
        # total number of constructors
        self.size = sum(len(head) for head in self._productions.values())

        # map entities to their ids
        self.prod2id = {prod: i for i, prod in enumerate(self.productions)}
        self.type2id = {type: i for i, type in enumerate(self.types)}
        self.field2id = {field: i for i, field in enumerate(self.fields)}
        self._name_field_map = {field.__repr__(plain=True): field for field in self.field2id}

        self.id2prod = {i: prod for i, prod in enumerate(self.productions)}
        self.id2type = {i: type for i, type in enumerate(self.types)}
        self.id2field = {i: field for i, field in enumerate(self.fields)}


    def __len__(self):
        return self.size


    @property
    def productions(self):
        return sorted(chain.from_iterable(self._productions.values()), key=lambda x: repr(x))


    def __getitem__(self, tp):
        if isinstance(tp, ASDLType):
            return self._productions[tp]
        raise ValueError('ASDLGrammar only accept keys of type ASDLType')


    def get_prod_by_name(self, name):
        return self._constructor_production_map[name]


    def get_field_by_name(self, type_name):
        return self._name_field_map[type_name]


    @property
    def types(self):
        if not hasattr(self, '_types'):
            all_types = set()
            for prod in self.productions:
                all_types.add(prod.type)
                all_types.update(map(lambda x: x.type, prod.constructor.fields))

            self._types = sorted(all_types, key=lambda x: x.name)

        return self._types


    @property
    def fields(self):
        if not hasattr(self, '_fields'):
            all_fields = set()
            for prod in self.productions:
                all_fields.update(prod.constructor.fields.keys())

            self._fields = sorted(all_fields, key=lambda x: (x.name, x.type.name))

        return self._fields


    @property
    def primitive_types(self):
        return filter(lambda x: isinstance(x, ASDLPrimitiveType), self.types)


    @property
    def composite_types(self):
        return filter(lambda x: isinstance(x, ASDLCompositeType), self.types)


    def is_composite_type(self, asdl_type):
        return asdl_type in self.composite_types


    def is_primitive_type(self, asdl_type):
        return asdl_type in self.primitive_types


    @staticmethod
    def from_filepath(file_path):

        def _parse_field_from_text(_text):
            d = _text.strip().split(' ')
            _type, _name = d[0].strip(), d[1].strip()
            if _type in primitive_type_names:
                return Field(_name, ASDLPrimitiveType(_type))
            else: return Field(_name, ASDLCompositeType(_type))

        def _parse_constructor_from_text(_text):
            _text = _text.strip()
            if '(' in _text:
                lidx, ridx = _text.find('('), _text.find(')')
                name, field_blocks = _text[:lidx], _text[lidx + 1:ridx].split(',')
                field_list = list(map(_parse_field_from_text, field_blocks))
                if '[' in name: # enumerable production rules
                    num_constraints = re.search(r'\[(.*?)\]', name).group(1)
                    name = name[:name.find('[')]
                    field, num_constraints = num_constraints.split(':')
                    field = _parse_field_from_text(field)
                    _min, _max = num_constraints.split(',')
                    _min = int(_min.strip()) if _min.strip() else 1
                    _max = int(_max.strip()) if _max.strip() else len(ASDLConstructor.number2word) - 1
                    constructor_list = []
                    for number in range(_min, _max + 1):
                        field_dict = OrderedDict()
                        for f in field_list:
                            if f not in field_dict:
                                field_dict[f] = 0
                            field_dict[f] += 1
                        field_dict[field] = number # over-write the fields definition in the constructor
                        constructor = ASDLConstructor(name, field_dict, number=number)
                        constructor_list.append(constructor)
                else:
                    field_dict = OrderedDict()
                    for field in field_list: # following the left-to-right order in grammar definition
                        if field not in field_dict:
                            field_dict[field] = 0
                        field_dict[field] += 1
                    constructor_list = [ASDLConstructor(name, field_dict)]
                return constructor_list
            else: # no fields
                return [ASDLConstructor(_text)]


        with open(file_path, 'r') as inf:
            text = inf.read()
        lines = remove_comment(text).split('\n')
        lines = list(filter(lambda l: l.strip(), lines))
        line_no = 0

        # first line is always the primitive types
        primitive_type_names = list(map(lambda x: x.strip(), lines[line_no].split(',')))
        line_no += 1

        all_productions = list()
        while True:
            type_block = lines[line_no]
            type_name = type_block[:type_block.find('=')].strip()
            constructors_blocks = type_block[type_block.find('=') + 1:].split('|')
            i = line_no + 1

            while i < len(lines) and lines[i].strip().startswith('|'):
                t = lines[i].strip()
                cont_constructors_blocks = t[1:].split('|')
                constructors_blocks.extend(cont_constructors_blocks)

                i += 1

            constructors_blocks = filter(lambda x: x and x.strip(), constructors_blocks)

            # parse type name
            new_type = ASDLPrimitiveType(type_name) if type_name in primitive_type_names else ASDLCompositeType(type_name)
            constructors = sum(map(_parse_constructor_from_text, constructors_blocks), [])

            productions = list(map(lambda c: ASDLProduction(new_type, c), constructors))
            all_productions.extend(productions)

            line_no = i
            if line_no == len(lines):
                break

        grammar = ASDLGrammar(all_productions, file_path)
        return grammar


class ASDLProduction(object):

    def __init__(self, type, constructor):
        self.type = type
        self.constructor = constructor


    @property
    def fields(self):
        return self.constructor.fields


    def __getitem__(self, field):
        return self.constructor[field]


    def __hash__(self):
        h = hash(self.type) ^ hash(self.constructor)
        return h


    def __eq__(self, other):
        return isinstance(other, ASDLProduction) and \
               self.type == other.type and \
               self.constructor == other.constructor


    def __ne__(self, other):
        return not self.__eq__(other)


    def __repr__(self):
        return '%s -> %s' % (self.type.__repr__(plain=True), self.constructor.__repr__(plain=True))


class ASDLConstructor(object):

    number2word = ('', 'One', 'Two', 'Three', 'Four', 'Five', 'Six',
        'Seven', 'Eight', 'Nine', 'Ten')

    def __init__(self, name='', fields={}, number=0):
        self.name = name + ASDLConstructor.number2word[number]
        self.fields = fields


    def __getitem__(self, field):
        return self.fields[field]


    def __hash__(self):
        h = hash(self.name)
        for field in self.fields:
            h ^= hash(field)
            h = h + 37 * hash(self.fields[field])
        return h


    def __eq__(self, other):
        return isinstance(other, ASDLConstructor) and \
            self.name == other.name and \
            self.fields == other.fields


    def __ne__(self, other):
        return not self.__eq__(other)


    def __repr__(self, plain=False):
        plain_repr = '%s(%s)' % (self.name,
                                 ', '.join(f.__repr__(plain=True) for f in self.fields for _ in range(self.fields[f])))
        if plain: return plain_repr
        else: return 'Constructor(%s)' % plain_repr


class Field(object):

    def __init__(self, name, type):
        self.name = name
        self.type = type


    def __hash__(self):
        h = hash(self.name) ^ hash(self.type)
        return h


    def __eq__(self, other):
        return type(other) == Field and \
               self.name == other.name and \
               self.type == other.type


    def __ne__(self, other):
        return not self.__eq__(other)


    def __repr__(self, plain=False):
        plain_repr = '%s %s' % (self.type.__repr__(plain=True), self.name)
        if plain: return plain_repr
        else: return 'Field(%s)' % plain_repr


    def __str__(self):
        return self.__repr__(plain=True)


    def copy(self):
        return Field(self.name, self.type.copy())


class ASDLType(object):

    def __init__(self, type_name):
        self.name = type_name


    def __hash__(self):
        return hash(self.name)


    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name


    def __ne__(self, other):
        return not self.__eq__(other)


    def __repr__(self, plain=False):
        plain_repr = self.name
        if plain: return plain_repr
        else: return '%s(%s)' % (self.__class__.__name__, plain_repr)


    def copy(self):
        return type(self)(self.name)


class ASDLCompositeType(ASDLType):

    pass


class ASDLPrimitiveType(ASDLType):

    pass
