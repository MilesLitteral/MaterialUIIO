# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: TextureDefinition.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor       as _descriptor
from google.protobuf import descriptor_pool  as _descriptor_pool
from google.protobuf import symbol_database  as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17TextureDefinition.proto\"a\n\x15RecommendationRequest\x12\x0f\n\x07user_id\x18\x01 \x01(\x05\x12\"\n\x08\x63\x61tegory\x18\x02 \x01(\x0e\x32\x10.TextureCategory\x12\x13\n\x0bmax_results\x18\x03 \x01(\x05\"2\n\x15TextureRecommendation\x12\n\n\x02id\x18\x01 \x01(\x05\x12\r\n\x05title\x18\x02 \x01(\t\"I\n\x16RecommendationResponse\x12/\n\x0frecommendations\x18\x01 \x03(\x0b\x32\x16.TextureRecommendation*5\n\x0fTextureCategory\x12\x0e\n\nMONOCHROME\x10\x00\x12\t\n\x05\x43OLOR\x10\x01\x12\x07\n\x03XYZ\x10\x02\x32O\n\x0fRecommendations\x12<\n\tRecommend\x12\x16.RecommendationRequest\x1a\x17.RecommendationResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'TextureDefinition_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_TEXTURECATEGORY']._serialized_start=253
  _globals['_TEXTURECATEGORY']._serialized_end=306
  _globals['_RECOMMENDATIONREQUEST']._serialized_start=27
  _globals['_RECOMMENDATIONREQUEST']._serialized_end=124
  _globals['_TEXTURERECOMMENDATION']._serialized_start=126
  _globals['_TEXTURERECOMMENDATION']._serialized_end=176
  _globals['_RECOMMENDATIONRESPONSE']._serialized_start=178
  _globals['_RECOMMENDATIONRESPONSE']._serialized_end=251
  _globals['_RECOMMENDATIONS']._serialized_start=308
  _globals['_RECOMMENDATIONS']._serialized_end=387
# @@protoc_insertion_point(module_scope)
