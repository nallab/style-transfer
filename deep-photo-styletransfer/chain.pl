#!/usr/bin/env perl
use strict;
use warnings;

# 環境変数に指定されている各画像ファイルのパスを読み込む
my $content     = $ENV{CONTENT};
my $style       = $ENV{STYLE};
# my $content_seg = $ENV{CONTENTSEG};
# my $style_seg   = $ENV{STYLESEG};

# 各画像のファイル名を取得する
chdir($content);
my @cnt_files = glob "*.png";
chdir($style);
my @sty_files = glob "*.png";
# chdir($content_seg);
# my @cnt_seg_files = glob "*.png";
# chdir($style_seg);
# my @sty_seg_files = glob "*.png";

my $output = "";

foreach my $c(@cnt_files) {
    foreach my $s(@sty_files) {
	$output .=
	$c . " " .
	$c . " " .
	$s . " " .
	$s . "\n";
    }
}

$output =~ s/(^|\n)(HD-\d+\.png\s?)(HD-\d+?)(\.png\s?)(HD-\d+\.png\s?)(HD-\d+?)(\.png?)/$1$2$3-seg$4$5$6-seg$7/g;
# $output =~ s/(^|\n)\sHD.*\s(HD-.*)\sHD-.*\s(HD-.*)(\n|$)/a/g;

print $output;
