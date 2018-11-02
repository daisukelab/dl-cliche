import openpyxl as opx
from copy import copy

def opx_copy_cell(source_cell, target_cell, style_copy=False):
    """openpyxl helper: Copy cell from source to target.
    Cells can be on different sheets.

    Args:
        style_copy: Flag to copy style or not.
    """
    target_cell._value = source_cell._value
    target_cell.data_type = source_cell.data_type
    if style_copy and source_cell.has_style:
        target_cell._style = copy(source_cell._style)
    if source_cell.hyperlink:
        target_cell._hyperlink = copy(source_cell.hyperlink)
    if source_cell.comment:
        target_cell.comment = copy(source_cell.comment)

def opx_copy_cell_style(source_cell, target_cell):
    """openpyxl helper: Copy cell style only from source to target.
    """
    target_cell._style = copy(source_cell._style)

def opx_copy_row(sh_src, row_src, sh_dest, row_dest, n_col=None, style_copy=False, debug=False):
    """openpyxl helper: Copy row.

    Args:
        sh_src: Source sheet.
        row_src: Source row number, range is as usual: `[0, max-1]`.
        sh_dest: Destination sheet.
        row_dest: Destination row number, same range as source.
        n_col: Number of columns to copy, n_col=5 will copy five sequential columns.
        style_copy: Flag to copy style or not.
    """
    if n_col is None: n_col = sh_src.max_column
    if debug:
        print('<', 1, row_src+1, sh_src.cell(column=1, row=row_src+1).value)
        print('>', 1, row_dest+1, sh_dest.cell(column=1, row=row_dest+1).value)
    for c in range(n_col):
        opx_copy_cell(sh_src.cell(column=c+1, row=row_src+1), sh_dest.cell(column=c+1, row=row_dest+1),
                     style_copy=style_copy)

def opx_duplicate_style(sh, row_src, row_dest, n_row=None, debug=False):
    """openpyxl helper: Copy style among rows.

    Args:
        sh: Sheet.
        row_src: Source row number, range is as usual: `[0, max-1]`.
        row_dest: Destination row number, same range as source.
        n_row: Number of rows to copy, n_col=5 will copy five sequential rows.
        style_copy: Flag to copy style or not.
    """
    n_col = sh.max_column
    if n_row is None:
        n_row = sh.max_row - row_dest
    if debug:
        print('duplicate', row_src, 'style to', row_dest, '-', row_dest+n_row - 1)
    for r in range(row_dest, row_dest+n_row):
        for c in range(n_col):
            opx_copy_cell_style(sh.cell(column=c+1, row=row_src+1), sh.cell(column=c+1, row=r+1))
