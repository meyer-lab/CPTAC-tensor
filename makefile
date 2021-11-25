.PHONY: clean test

flist = 2
flistFull = $(patsubst %, output/figure%.svg, $(flist))

all: test $(flistFull)

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

output/figure%.svg: genFigures.py cpactensor/figures/figure%.py venv
	@ mkdir -p ./output
	. venv/bin/activate && ./genFigures.py $*

test: venv
	. venv/bin/activate && pytest -s -v -x

clean:
	rm -rf venv