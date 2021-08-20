flist = 1 2

all: $(patsubst %, figure%.svg, $(flist))

# Figure rules
figure%.svg: venv genFigure.py cptactensor/figures/figure%.py
. venv/bin/activate && ./genFigure.py $*

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

output/manuscript.md: venv manuscripts/*.md
	mkdir -p ./output
	. venv/bin/activate && manubot process --content-directory=manuscripts/ --output-directory=output/ --cache-directory=cache --skip-citations --log-level=INFO
	git remote rm rootstock

output/manuscript.html: venv output/manuscript.md $(patsubst %, figure%.svg, $(flist))
	cp *.svg output/
	. venv/bin/activate && pandoc --verbose \
		--defaults=common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml \
		--output=output/manuscript.html output/manuscript.md

output/manuscript.docx: venv output/manuscript.md $(patsubst %, figure%.svg, $(flist))
	cp *.svg output/
	. venv/bin/activate && pandoc --verbose \
		--defaults=common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml \
		--output=output/manuscript.docx output/manuscript.md

test: venv
	. venv/bin/activate && pytest -s -v -x

testcover: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=cptactensor --cov-report xml:coverage.xml

clean:
	rm -rf *.pdf venv pylint.log figure*.svg
	git checkout HEAD -- output
	git clean -ffdx output
