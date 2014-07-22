default: html

html: clean
	pelican content

clean:
	rm -rf cache

.PHONY: html
