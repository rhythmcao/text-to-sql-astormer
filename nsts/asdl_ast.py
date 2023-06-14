#coding=utf-8
from io import StringIO
from collections import OrderedDict
from typing import List, Dict
from nsts.asdl import ASDLCompositeType, Field, ASDLProduction


class RealizedField(Field):
    """ Wrapper of Field object with realized values
    """
    def __init__(self, field: Field, value = None, realized_time: int = 0, parent_node = None) -> None:
        super(RealizedField, self).__init__(field.name, field.type)
        self.field: Field = field
        self.value: List[AbstractSyntaxTree, int, List[int]] = value
        self.realized_time: int = realized_time
        self.parent_node: AbstractSyntaxTree = parent_node


    def add_value(self, value, realized_time: int = 0):
        if isinstance(value, AbstractSyntaxTree):
            value.parent_field = self
        self.value = value
        self.realized_time = realized_time


    @property
    def depth(self):
        return self.parent_node.depth + 1


    @property
    def finished(self):
        if self.value is None: return False
        elif isinstance(self.value, AbstractSyntaxTree):
            return self.value.finished
        else: return True


class AbstractSyntaxTree(object):

    def __init__(self, production: ASDLProduction, created_time: int = 0, depth: int = 0, parent_field: RealizedField = None) -> None:
        self.production: ASDLProduction = production
        self.fields: Dict[Field, List[RealizedField]] = OrderedDict()
        for field in self.production.fields:
            for _ in range(self.production.fields[field]):
                self._add_child(RealizedField(field))

        self.created_time: int = created_time # used in decoding
        self.depth: int = depth # depth of the current node
        self.parent_field: RealizedField = parent_field

        # record the order of generation for RealizedField
        self.tracker = {field: 0 for field in self.production.fields}


    def _add_child(self, realized_field: RealizedField):
        if realized_field.field not in self.fields:
            self.fields[realized_field.field] = []
        self.fields[realized_field.field].append(realized_field)
        realized_field.parent_node = self


    def __getitem__(self, field):
        return self.fields[field]


    def copy(self):
        new_tree = AbstractSyntaxTree(self.production, created_time=self.created_time, depth=self.depth)
        for field in self.fields:
            new_field_list = new_tree.fields[field]
            for idx, old_field in enumerate(self.fields[field]):
                new_field = new_field_list[idx]
                if old_field.value is not None:
                    if isinstance(old_field.type, ASDLCompositeType):
                        new_field.add_value(old_field.value.copy(), realized_time=old_field.realized_time)
                    else: new_field.add_value(old_field.value, realized_time=old_field.realized_time)
        new_tree.tracker = {k: v for k, v in self.tracker.items()}
        return new_tree


    def to_string(self, sb: StringIO = None, indent: int = 0, tables: list = None, columns: list = None, tokenizer = None) -> str:
        is_root = False
        if sb is None:
            is_root = True
            sb = StringIO()
            sb.write('sql-root := ')
            # sb.write('{\\tt sql} $\\rightarrow$ ') # to_latex_code

        sb.write('Node[j=%d, %s]\n' % (self.created_time, self.production.__repr__()))
        # sb.write('Node[ $j=%d$, %s ]\\\\\n' % (self.created_time, self.production.to_latex_code()))

        field_with_timestep = sorted([(field, min([rf.realized_time for rf in self.fields[field]])) for field in self.fields], key=lambda x: x[1])
        indent += 4
        for field, _ in field_with_timestep:
            rfs = sorted(self.fields[field], key=lambda x: x.realized_time)
            if isinstance(field.type, ASDLCompositeType): # sub-trees
                for rf in rfs:
                    prefix = ' ' * indent + '%s-%s := ' % (rf.type.name, rf.name)
                    # prefix = '\qquad' * (indent // 4) + ' %s $\\rightarrow$ ' % (rf.type.to_latex_code())
                    sb.write(prefix)
                    if rf.value is not None:
                        rf.value.to_string(sb, indent, tables, columns, tokenizer)
                    else: sb.write('?\n')
            else: # primitive types
                for rf in rfs:
                    if rf.value is not None:
                        if rf.type.name == 'tab_id':
                            val = tables[int(rf.value)] if tables is not None else str(rf.value)
                            # value = 'Leaf[ $j=%d$, {\\tt tab\_id} := %s ]' % (rf.realized_time, val) # to_latex_code
                        elif rf.type.name == 'col_id':
                            if int(rf.value) == 0: val = '*'
                            else:
                                tab = tables[columns[int(rf.value)][0]] + '.' if tables is not None and columns is not None else ''
                                col = columns[int(rf.value)][1] if columns is not None else str(rf.value)
                                val = tab + col
                            # value = 'Leaf[ $j=%d$, {\\tt col\_id} := %s ]' % (rf.realized_time, val) # to_latex_code
                        else:
                            val = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(rf.value)) if tokenizer is not None else str(rf.value)
                            # value = "Leaf[ $j=%d$, {\\tt tok\_id} := ``%s'' ]" % (rf.realized_time, val) # to_latex_code
                        value = 'Leaf[j=%d, val=%s]' % (rf.realized_time, val)
                    else: value = '?'
                    serial = ' ' * indent + '%s-%s := %s\n' % (rf.type.name, rf.name, value)
                    # serial = '\qquad' * (indent // 4) + ' %s $\\rightarrow$ %s\\\\\n' % (rf.type.to_latex_code(), value) # to_latex_code
                    sb.write(serial)

        if is_root:
            return sb.getvalue().rstrip('\n')


    def __repr__(self):
        return repr(self.production)


    @property
    def size(self):
        node_num = 1
        for field in self.fields:
            for realized_field in self.fields[field]:
                value = realized_field.value
                if isinstance(value, AbstractSyntaxTree):
                    node_num += value.size
                elif type(value) == list:
                    node_num += len(value)
                else: node_num += 1
        return node_num


    def field_finished(self, field):
        return self.tracker[field] == self.production.fields[field]