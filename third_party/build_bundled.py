#!/usr/bin/env python3
import os


mydir = os.path.dirname(__file__)
licenses = {'LICENSE', 'LICENSE.txt', 'LICENSE.rst', 'COPYING.BSD'}


def collect_license(current):
    collected = {}
    for root, dirs, files in os.walk(current):
        license = list(licenses & set(files))
        if license:
            name = root.split('/')[-1]
            license_file = os.path.join(root, license[0])
            try:
                ident = identify_license(license_file)
            except ValueError:
                raise ValueError('could not identify license file '
                                 f'for {root}') from None
            val = {
                'Name': name,
                'Files': [root],
                'License': ident,
                'License_file': [license_file],
            }
            if name in collected:
                # Only add it if the license is different
                if collected[name]['License'] == ident:
                    collected[name]['Files'].append(root)
                    collected[name]['License_file'].append(license_file)
                else:
                    collected[name + f' ({root})'] = val
            else:
                collected[name] = val
    return collected


def create_bundled(d, outstream):
    """Write the information to an open outstream"""
    collected = collect_license(d)
    sorted_keys = sorted(collected.keys())
    outstream.write('The Pytorch repository and source distributions bundle '
                    'several libraries that are \n')
    outstream.write('compatibly licensed.  We list these here.\n\n')
    for k in sorted_keys:
        c = collected[k]
        files = ',\n     '.join(c['Files'])
        license_file = ',\n     '.join(c['License_file'])
        outstream.write(f"Name: {c['Name']}\n")
        outstream.write(f"License: {c['License']}\n")
        outstream.write(f"Files: {files}\n")
        outstream.write('  For details, see ')
        outstream.write(license_file)
        outstream.write('\n\n') 


def identify_license(f, exception=''):
    """
    Read f and try to identify the license type
    This is __very__ rough and probably not legally binding, it is specific for
    this repo.
    """
    def squeeze(t):
        """Remove 'n and ' ', normalize quotes
        """
        t = t.replace('\n', '').replace(' ', '')
        t = t.replace('``', '"').replace("''", '"')
        return t

    with open(f) as fid:
        txt = fid.read()
        if not exception and 'exception' in txt:
            license = identify_license(f, 'exception')
            return license + ' with exception'
        txt = squeeze(txt)
        if 'ApacheLicense' in txt:
            # Hmm, do we need to check the text?
            return 'Apache-2.0'
        elif 'MITLicense' in txt:
            # Hmm, do we need to check the text?
            return 'MIT'
        elif 'BSD-3-ClauseLicense' in txt:
            # Hmm, do we need to check the text?
            return 'BSD-3-Clause'
        elif 'BSD3-ClauseLicense' in txt:
            # Hmm, do we need to check the text?
            return 'BSD-3-Clause'
        elif 'BoostSoftwareLicense-Version1.0' in txt:
            # Hmm, do we need to check the text?
            return 'BSL-1.0'
        elif all([squeeze(m) in txt.lower() for m in bsd3_txt]):
            return 'BSD-3-Clause'
        elif all([squeeze(m) in txt.lower() for m in bsd3_v1_txt]):
            return 'BSD-3-Clause'
        elif all([squeeze(m) in txt.lower() for m in bsd2_txt]):
            return 'BSD-2-Clause'
        elif all([squeeze(m) in txt.lower() for m in bsd3_src_txt]):
            return 'BSD-Source-Code'
        elif all([squeeze(m) in txt.lower() for m in mit_txt]):
            return 'MIT'
        else:
            raise ValueError('unknown license')

mit_txt = ['permission is hereby granted, free of charge, to any person '
           'obtaining a copy of this software and associated documentation '
           'files (the "software"), to deal in the software without '
           'restriction, including without limitation the rights to use, copy, '
           'modify, merge, publish, distribute, sublicense, and/or sell copies '
           'of the software, and to permit persons to whom the software is '
           'furnished to do so, subject to the following conditions:',

           'the above copyright notice and this permission notice shall be '
           'included in all copies or substantial portions of the software.',

           'the software is provided "as is", without warranty of any kind, '
           'express or implied, including but not limited to the warranties of '
           'merchantability, fitness for a particular purpose and '
           'noninfringement. in no event shall the authors or copyright holders '
           'be liable for any claim, damages or other liability, whether in an '
           'action of contract, tort or otherwise, arising from, out of or in '
           'connection with the software or the use or other dealings in the '
           'software.'
           ]

bsd3_txt = ['redistribution and use in source and binary forms, with or without '
            'modification, are permitted provided that the following conditions '
            'are met:',

            'redistributions of source code',

            'redistributions in binary form',

            'neither the name',

            'this software is provided by the copyright holders and '
            'contributors "as is" and any express or implied warranties, '
            'including, but not limited to, the implied warranties of '
            'merchantability and fitness for a particular purpose are disclaimed.',
            ]

# BSD2 is BSD3 without the "neither the name..." clause
bsd2_txt = bsd3_txt[:3] + bsd3_txt[4:]

# This BSD3 variant leaves "and contributors" out of the last clause of BSD-3,
# which is still valid BSD-3
v1 = bsd3_txt[4].replace('and contributors', '')
bsd3_v1_txt = bsd3_txt[:3] + [v1]

# This source variant of BSD-3 leaves the "redistributions in binary form" out
# which is https://spdx.org/licenses/BSD-Source-Code.html
bsd3_src_txt = bsd3_txt[:2] + bsd3_txt[4:]


if __name__ == '__main__':
    third_party = os.path.join(mydir)
    fname = os.path.join(third_party, 'LICENSES_BUNDLED.txt')
    with open(fname, 'w') as fid:
        create_bundled(third_party, fid)
